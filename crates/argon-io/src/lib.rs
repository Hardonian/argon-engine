//! Hardware-in-the-loop transport traits, streaming persistence, and an emulated device.

#![forbid(non_ascii_idents)]

use argon_runtime::{ArgonCoordinator, RuntimeError, TickTelemetry, TraceRecord};
use std::fmt;
use std::io::{Read, Write};

/// Magic header identifier for Argon binary telemetry log files.
pub const LOG_MAGIC: &[u8; 8] = b"ARGONLOG";
/// Log format version.
pub const LOG_VERSION: u32 = 1;
/// Fixed size of the binary log header in bytes.
pub const LOG_HEADER_SIZE: usize = 16;
/// Fixed byte size of a serialized [`TraceRecord`].
pub const RECORD_SIZE: usize = 24;

/// Errors produced by a physical or emulated device adapter or telemetry stream.
#[derive(Debug)]
pub enum IoError {
    /// Device and coordinator dimensions are incompatible.
    DimensionMismatch,
    /// The device transport failed.
    Transport(&'static str),
    /// Neural execution failed.
    Runtime(RuntimeError),
    /// Underlying I/O stream failure.
    Io(std::io::Error),
    /// Corrupted or invalid binary log file format.
    CorruptedLog(&'static str),
}

impl fmt::Display for IoError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch => write!(formatter, "device dimensions are incompatible"),
            Self::Transport(message) => write!(formatter, "device transport failed: {message}"),
            Self::Runtime(error) => write!(formatter, "device runtime failed: {error}"),
            Self::Io(error) => write!(formatter, "io persistence error: {error}"),
            Self::CorruptedLog(reason) => write!(formatter, "corrupted telemetry log: {reason}"),
        }
    }
}

impl std::error::Error for IoError {}

impl From<RuntimeError> for IoError {
    fn from(error: RuntimeError) -> Self {
        Self::Runtime(error)
    }
}

impl From<std::io::Error> for IoError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

/// Synchronous fixed-dimension physical device contract.
pub trait PhysicalDevice {
    /// Number of sensory and action channels.
    fn dimension(&self) -> usize;

    /// Writes one current sensor sample into caller-owned storage.
    ///
    /// # Errors
    /// Returns [`IoError`] when the transport or output buffer is invalid.
    fn read_sensors(&mut self, output: &mut [f32]) -> Result<(), IoError>;

    /// Applies one caller-owned actuator vector.
    ///
    /// # Errors
    /// Returns [`IoError`] when the transport or input buffer is invalid.
    fn write_actuators(&mut self, actions: &[f32]) -> Result<(), IoError>;
}

/// Streaming binary persistent writer for [`TraceRecord`] streams.
#[derive(Debug)]
pub struct TelemetryLogWriter<W> {
    writer: W,
    records_written: u64,
}

impl<W: Write> TelemetryLogWriter<W> {
    /// Creates a new writer and writes the 16-byte magic header.
    ///
    /// # Errors
    /// Returns [`IoError`] if writing the header fails.
    pub fn new(mut writer: W) -> Result<Self, IoError> {
        let mut header = [0_u8; LOG_HEADER_SIZE];
        header[0..8].copy_from_slice(LOG_MAGIC);
        header[8..12].copy_from_slice(&LOG_VERSION.to_le_bytes());
        header[12..16].copy_from_slice(&0_u32.to_le_bytes());
        writer.write_all(&header)?;
        Ok(Self {
            writer,
            records_written: 0,
        })
    }

    /// Appends one [`TraceRecord`] to the persistent stream with zero heap allocations.
    ///
    /// # Errors
    /// Returns [`IoError`] if serialization or writing fails.
    pub fn write_record(&mut self, record: &TraceRecord) -> Result<(), IoError> {
        let mut buf = [0_u8; RECORD_SIZE];
        buf[0..8].copy_from_slice(&record.sequence_id.to_le_bytes());
        buf[8..12].copy_from_slice(&record.mean_goodness.to_le_bytes());
        buf[12..16].copy_from_slice(&record.output_l2_norm.to_le_bytes());
        buf[16..20].copy_from_slice(&record.queue_depth.to_le_bytes());
        buf[20..24].copy_from_slice(&record.flags.to_le_bytes());
        self.writer.write_all(&buf)?;
        self.records_written = self.records_written.saturating_add(1);
        Ok(())
    }

    /// Flushes any buffered bytes to underlying storage.
    ///
    /// # Errors
    /// Returns [`IoError`] if flushing fails.
    pub fn flush(&mut self) -> Result<(), IoError> {
        self.writer.flush()?;
        Ok(())
    }

    /// Appends multiple [`TraceRecord`]s to the persistent stream in a tight loop.
    ///
    /// # Errors
    /// Returns [`IoError`] if serialization or writing fails.
    pub fn write_records(&mut self, records: &[TraceRecord]) -> Result<(), IoError> {
        for record in records {
            self.write_record(record)?;
        }
        Ok(())
    }

    /// Number of records written in this stream.
    pub const fn records_written(&self) -> u64 {
        self.records_written
    }

    /// Consumes the writer and returns the underlying sink.
    pub fn into_inner(self) -> W {
        self.writer
    }
}

/// Binary persistent reader for replay and verification of [`TraceRecord`] streams.
#[derive(Debug)]
pub struct TelemetryLogReader<R> {
    reader: R,
    records_read: u64,
}

impl<R: Read> TelemetryLogReader<R> {
    /// Opens and validates the telemetry log header.
    ///
    /// # Errors
    /// Returns [`IoError::CorruptedLog`] if magic or version is invalid, or [`IoError::Io`] on read failure.
    pub fn new(mut reader: R) -> Result<Self, IoError> {
        let mut header = [0_u8; LOG_HEADER_SIZE];
        reader.read_exact(&mut header)?;
        if &header[0..8] != LOG_MAGIC {
            return Err(IoError::CorruptedLog(
                "invalid argon telemetry magic header",
            ));
        }
        let version = u32::from_le_bytes([header[8], header[9], header[10], header[11]]);
        if version != LOG_VERSION {
            return Err(IoError::CorruptedLog("unsupported telemetry log version"));
        }
        Ok(Self {
            reader,
            records_read: 0,
        })
    }

    /// Reads the next [`TraceRecord`], returning `Ok(None)` on clean EOF.
    ///
    /// # Errors
    /// Returns [`IoError::CorruptedLog`] on unexpected truncated record or [`IoError::Io`] on read failure.
    pub fn read_record(&mut self) -> Result<Option<TraceRecord>, IoError> {
        let mut buf = [0_u8; RECORD_SIZE];
        let mut total_read = 0;
        while total_read < RECORD_SIZE {
            match self.reader.read(&mut buf[total_read..]) {
                Ok(0) => break,
                Ok(n) => total_read += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(IoError::Io(e)),
            }
        }
        if total_read == 0 {
            return Ok(None);
        }
        if total_read < RECORD_SIZE {
            return Err(IoError::CorruptedLog("truncated telemetry record at EOF"));
        }
        let sequence_id = u64::from_le_bytes([
            buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
        ]);
        let mean_goodness = f32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
        let output_l2_norm = f32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]);
        let queue_depth = u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]);
        let flags = u32::from_le_bytes([buf[20], buf[21], buf[22], buf[23]]);

        self.records_read = self.records_read.saturating_add(1);
        Ok(Some(TraceRecord {
            sequence_id,
            mean_goodness,
            output_l2_norm,
            queue_depth,
            flags,
        }))
    }

    /// Number of records read so far.
    pub const fn records_read(&self) -> u64 {
        self.records_read
    }

    /// Reads up to `destination.len()` records into caller-allocated storage.
    ///
    /// Returns the number of records read.
    ///
    /// # Errors
    /// Returns [`IoError`] on corrupted log format or I/O failure.
    pub fn read_batch(&mut self, destination: &mut [TraceRecord]) -> Result<usize, IoError> {
        let mut count = 0;
        for slot in destination.iter_mut() {
            match self.read_record()? {
                Some(record) => {
                    *slot = record;
                    count += 1;
                }
                None => break,
            }
        }
        Ok(count)
    }

    /// Consumes the reader and returns the underlying stream.
    pub fn into_inner(self) -> R {
        self.reader
    }
}

/// Preallocated neural loop around a physical device adapter.
#[derive(Debug)]
pub struct DeviceLoop<D> {
    device: D,
    sensors: Vec<f32>,
    actions: Vec<f32>,
    sequence_id: u64,
}

impl<D: PhysicalDevice> DeviceLoop<D> {
    /// Creates a loop after checking device/coordinator dimensions.
    ///
    /// # Errors
    /// Returns [`IoError::DimensionMismatch`] for incompatible dimensions.
    pub fn new(device: D, coordinator: &ArgonCoordinator) -> Result<Self, IoError> {
        let dimension = device.dimension();
        if dimension == 0 || dimension != coordinator.tile_dim() {
            return Err(IoError::DimensionMismatch);
        }
        Ok(Self {
            device,
            sensors: vec![0.0; dimension],
            actions: vec![0.0; dimension],
            sequence_id: 0,
        })
    }

    /// Executes one allocation-free hardware sensory-control tick.
    ///
    /// # Errors
    /// Returns [`IoError`] for device or neural failures.
    pub fn tick(&mut self, coordinator: &mut ArgonCoordinator) -> Result<TickTelemetry, IoError> {
        self.device.read_sensors(&mut self.sensors)?;
        let telemetry =
            coordinator.step_borrowed(self.sequence_id, 1.0, &self.sensors, &mut self.actions)?;
        self.device.write_actuators(&self.actions)?;
        self.sequence_id = self.sequence_id.wrapping_add(1);
        Ok(telemetry)
    }

    /// Executes one hardware tick and appends the resulting trace record to a persistent telemetry log.
    ///
    /// # Errors
    /// Returns [`IoError`] for device, coordinator, or I/O persistence failures.
    pub fn tick_and_log<W: Write>(
        &mut self,
        coordinator: &mut ArgonCoordinator,
        log_writer: &mut TelemetryLogWriter<W>,
    ) -> Result<TickTelemetry, IoError> {
        let telemetry = self.tick(coordinator)?;
        let record = TraceRecord {
            sequence_id: telemetry.sequence_id,
            mean_goodness: telemetry.mean_goodness,
            output_l2_norm: telemetry.output_l2_norm,
            queue_depth: 0,
            flags: 0,
        };
        log_writer.write_record(&record)?;
        Ok(telemetry)
    }

    /// Returns the underlying adapter.
    pub const fn device(&self) -> &D {
        &self.device
    }
}

/// Deterministic first-order emulated device for CI and integration development.
#[derive(Debug, Clone)]
pub struct LoopbackDevice {
    state: Vec<f32>,
    response: f32,
}

impl LoopbackDevice {
    /// Creates an emulated device with a response coefficient in `(0, 1]`.
    ///
    /// # Errors
    /// Returns [`IoError`] when dimensions or response are invalid.
    pub fn new(dimension: usize, response: f32) -> Result<Self, IoError> {
        if dimension == 0 || !response.is_finite() || response <= 0.0 || response > 1.0 {
            return Err(IoError::Transport(
                "loopback dimension and response are invalid",
            ));
        }
        Ok(Self {
            state: vec![0.25; dimension],
            response,
        })
    }

    /// Returns the current emulated sensor state.
    pub fn state(&self) -> &[f32] {
        &self.state
    }
}

impl PhysicalDevice for LoopbackDevice {
    fn dimension(&self) -> usize {
        self.state.len()
    }

    fn read_sensors(&mut self, output: &mut [f32]) -> Result<(), IoError> {
        if output.len() != self.state.len() {
            return Err(IoError::DimensionMismatch);
        }
        output.copy_from_slice(&self.state);
        Ok(())
    }

    fn write_actuators(&mut self, actions: &[f32]) -> Result<(), IoError> {
        if actions.len() != self.state.len() || actions.iter().any(|value| !value.is_finite()) {
            return Err(IoError::DimensionMismatch);
        }
        for (state, &action) in self.state.iter_mut().zip(actions) {
            *state = self.response.mul_add(action - *state, *state);
        }
        Ok(())
    }
}

/// Fixed-capacity timestamped sensor frame representation for real-time buses (CAN-FD, Micro-ROS).
#[derive(Debug, Clone, PartialEq)]
pub struct SensorFrame<const N: usize> {
    /// Timestamp in nanoseconds.
    pub timestamp_ns: u64,
    /// Sequence identifier on the communication bus.
    pub sequence_id: u64,
    /// Hardware channel or sensor node identifier.
    pub channel_id: u32,
    /// Sensor readings payload.
    pub data: [f32; N],
}

impl<const N: usize> SensorFrame<N> {
    /// Creates a new sensor frame with the specified header and data.
    pub const fn new(timestamp_ns: u64, sequence_id: u64, channel_id: u32, data: [f32; N]) -> Self {
        Self {
            timestamp_ns,
            sequence_id,
            channel_id,
            data,
        }
    }
}

/// In-memory shared memory / IPC device adapter simulating zero-allocation bus exchange.
#[derive(Debug)]
pub struct SharedMemoryDevice {
    dimension: usize,
    sensor_buffer: Vec<f32>,
    actuator_buffer: Vec<f32>,
    exchange_count: u64,
}

impl SharedMemoryDevice {
    /// Creates a new shared-memory device with the given dimension.
    ///
    /// # Errors
    /// Returns [`IoError::Transport`] if dimension is zero.
    pub fn new(dimension: usize) -> Result<Self, IoError> {
        if dimension == 0 {
            return Err(IoError::Transport("shared memory dimension must be non-zero"));
        }
        Ok(Self {
            dimension,
            sensor_buffer: vec![0.5; dimension],
            actuator_buffer: vec![0.0; dimension],
            exchange_count: 0,
        })
    }

    /// Injects sensory data into the shared memory area.
    ///
    /// # Errors
    /// Returns [`IoError::DimensionMismatch`] if input length differs from dimension.
    pub fn write_sensor_bus(&mut self, sensors: &[f32]) -> Result<(), IoError> {
        if sensors.len() != self.dimension {
            return Err(IoError::DimensionMismatch);
        }
        self.sensor_buffer.copy_from_slice(sensors);
        Ok(())
    }

    /// Reads back actuator command states written by the coordinator.
    pub fn read_actuator_bus(&self) -> &[f32] {
        &self.actuator_buffer
    }

    /// Total exchange cycles completed.
    pub const fn exchange_count(&self) -> u64 {
        self.exchange_count
    }
}

impl PhysicalDevice for SharedMemoryDevice {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn read_sensors(&mut self, output: &mut [f32]) -> Result<(), IoError> {
        if output.len() != self.dimension {
            return Err(IoError::DimensionMismatch);
        }
        output.copy_from_slice(&self.sensor_buffer);
        self.exchange_count = self.exchange_count.saturating_add(1);
        Ok(())
    }

    fn write_actuators(&mut self, actions: &[f32]) -> Result<(), IoError> {
        if actions.len() != self.dimension {
            return Err(IoError::DimensionMismatch);
        }
        self.actuator_buffer.copy_from_slice(actions);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use argon_kernel::{PersistentGridConfig, PlasticityConfig};
    use std::io::Cursor;

    #[test]
    fn emulated_device_executes_hardware_loop() -> Result<(), Box<dyn std::error::Error>> {
        let grid = PersistentGridConfig {
            grid_dim_x: 1,
            block_dim_x: 64,
            shared_mem_bytes: 16_384,
            tile_dim: 8,
        };
        let plasticity = PlasticityConfig {
            learning_rate: 0.01,
            goodness_threshold: 1.0,
            weight_decay: 0.001,
            phase_polarity: 1.0,
        };
        let mut coordinator = ArgonCoordinator::new(2, &grid, plasticity, 4)?;
        let device = LoopbackDevice::new(8, 0.2)?;
        let mut device_loop = DeviceLoop::new(device, &coordinator)?;
        for _ in 0..100 {
            let telemetry = device_loop.tick(&mut coordinator)?;
            assert!(telemetry.mean_goodness.is_finite());
        }
        assert!(device_loop
            .device()
            .state()
            .iter()
            .all(|value| value.is_finite()));
        Ok(())
    }

    #[test]
    fn telemetry_streaming_log_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
        let mut sink = Vec::new();
        let mut writer = TelemetryLogWriter::new(&mut sink)?;

        let mut expected_records = Vec::new();
        for i in 0_u16..10_u16 {
            let record = TraceRecord {
                sequence_id: u64::from(i),
                mean_goodness: f32::from(i) * 0.1,
                output_l2_norm: 1.5,
                queue_depth: u32::from(i),
                flags: 0,
            };
            expected_records.push(record);
            writer.write_record(&record)?;
        }
        writer.flush()?;
        assert_eq!(writer.records_written(), 10);

        let mut reader = TelemetryLogReader::new(Cursor::new(sink))?;
        let mut recovered = Vec::new();
        while let Some(record) = reader.read_record()? {
            recovered.push(record);
        }
        assert_eq!(reader.records_read(), 10);
        assert_eq!(recovered, expected_records);
        Ok(())
    }

    #[test]
    fn device_loop_records_to_telemetry_stream() -> Result<(), Box<dyn std::error::Error>> {
        let grid = PersistentGridConfig {
            grid_dim_x: 1,
            block_dim_x: 64,
            shared_mem_bytes: 16_384,
            tile_dim: 8,
        };
        let plasticity = PlasticityConfig {
            learning_rate: 0.01,
            goodness_threshold: 1.0,
            weight_decay: 0.001,
            phase_polarity: 1.0,
        };
        let mut coordinator = ArgonCoordinator::new(2, &grid, plasticity, 4)?;
        let device = LoopbackDevice::new(8, 0.2)?;
        let mut device_loop = DeviceLoop::new(device, &coordinator)?;

        let mut sink = Vec::new();
        let mut writer = TelemetryLogWriter::new(&mut sink)?;

        for _ in 0..25 {
            let _ = device_loop.tick_and_log(&mut coordinator, &mut writer)?;
        }
        writer.flush()?;
        assert_eq!(writer.records_written(), 25);

        let mut reader = TelemetryLogReader::new(Cursor::new(sink))?;
        let mut count = 0;
        while let Some(rec) = reader.read_record()? {
            assert_eq!(rec.sequence_id, count);
            count += 1;
        }
        assert_eq!(count, 25);
        Ok(())
    }

    #[test]
    fn telemetry_corrupted_header_detected() {
        let corrupted_magic = b"NOTARGON\x01\x00\x00\x00\x00\x00\x00\x00";
        let res = TelemetryLogReader::new(Cursor::new(&corrupted_magic[..]));
        assert!(matches!(res, Err(IoError::CorruptedLog(_))));

        let wrong_version = b"ARGONLOG\x02\x00\x00\x00\x00\x00\x00\x00";
        let res_v = TelemetryLogReader::new(Cursor::new(&wrong_version[..]));
        assert!(matches!(res_v, Err(IoError::CorruptedLog(_))));
    }

    #[test]
    fn telemetry_truncated_record_detected() -> Result<(), Box<dyn std::error::Error>> {
        let mut sink = Vec::new();
        let mut writer = TelemetryLogWriter::new(&mut sink)?;
        writer.write_record(&TraceRecord {
            sequence_id: 1,
            mean_goodness: 0.5,
            output_l2_norm: 1.0,
            queue_depth: 0,
            flags: 0,
        })?;
        writer.flush()?;

        // Truncate the file mid-record
        sink.truncate(LOG_HEADER_SIZE + 10);
        let mut reader = TelemetryLogReader::new(Cursor::new(sink))?;
        let res = reader.read_record();
        assert!(matches!(res, Err(IoError::CorruptedLog(_))));
        Ok(())
    }

    #[test]
    fn telemetry_batch_write_and_read() -> Result<(), Box<dyn std::error::Error>> {
        let mut sink = Vec::new();
        let mut writer = TelemetryLogWriter::new(&mut sink)?;

        let batch = [
            TraceRecord {
                sequence_id: 10,
                mean_goodness: 0.1,
                output_l2_norm: 1.0,
                queue_depth: 0,
                flags: 0,
            },
            TraceRecord {
                sequence_id: 11,
                mean_goodness: 0.2,
                output_l2_norm: 1.1,
                queue_depth: 0,
                flags: 0,
            },
        ];

        writer.write_records(&batch)?;
        writer.flush()?;
        assert_eq!(writer.records_written(), 2);

        let mut reader = TelemetryLogReader::new(Cursor::new(sink))?;
        let mut dest = [TraceRecord {
            sequence_id: 0,
            mean_goodness: 0.0,
            output_l2_norm: 0.0,
            queue_depth: 0,
            flags: 0,
        }; 4];

        let read_count = reader.read_batch(&mut dest)?;
        assert_eq!(read_count, 2);
        assert_eq!(dest[0], batch[0]);
        assert_eq!(dest[1], batch[1]);
        Ok(())
    }

    #[test]
    fn shared_memory_device_and_sensor_frame_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
        let frame = SensorFrame::new(1_000_000, 42, 1, [0.1_f32, 0.2, 0.3, 0.4]);
        assert_eq!(frame.timestamp_ns, 1_000_000);
        assert_eq!(frame.sequence_id, 42);
        assert_eq!(frame.channel_id, 1);
        assert_eq!(frame.data.len(), 4);

        let mut shm = SharedMemoryDevice::new(4)?;
        assert_eq!(shm.dimension(), 4);
        assert_eq!(shm.exchange_count(), 0);

        shm.write_sensor_bus(&frame.data)?;
        let mut sensors_out = [0.0_f32; 4];
        shm.read_sensors(&mut sensors_out)?;
        assert_eq!(sensors_out, frame.data);
        assert_eq!(shm.exchange_count(), 1);

        let action_cmd = [0.8_f32, 0.7, 0.6, 0.5];
        shm.write_actuators(&action_cmd)?;
        assert_eq!(shm.read_actuator_bus(), &action_cmd);
        Ok(())
    }
}
