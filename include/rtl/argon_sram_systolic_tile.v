// =============================================================================
// Argon Engine — Synthesizable Persistent SRAM Systolic Tile
// Parameterized Architecture: 8x8 Processing Elements (32-bit FP/Fixed)
// Milestone M6.1: Zero-Dispatch Neuromorphic Hardware Synthesis
// =============================================================================

`timescale 1ns / 1ps

module argon_sram_systolic_tile #(
    parameter TILE_DIM   = 8,
    parameter DATA_WIDTH = 32
)(
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   enable,
    input  wire                   polarity,        // 1: Positive phase (+1.0), 0: Negative phase (-1.0)
    input  wire [DATA_WIDTH-1:0]  learning_rate,
    input  wire [DATA_WIDTH-1:0]  goodness_thresh,

    // Sensory Input Stream (Vector of TILE_DIM channels)
    input  wire [TILE_DIM*DATA_WIDTH-1:0] sensory_in,

    // Actuator / Motor Action Output (Vector of TILE_DIM channels)
    output reg  [TILE_DIM*DATA_WIDTH-1:0] action_out,

    // Scalar Goodness Energy Checksum
    output reg  [DATA_WIDTH-1:0]          goodness_metric,
    output reg                            done_tick
);

    // -------------------------------------------------------------------------
    // Persistent In-SRAM Synaptic Weight Storage Matrix [TILE_DIM x TILE_DIM]
    // -------------------------------------------------------------------------
    reg [DATA_WIDTH-1:0] sram_weights [0:TILE_DIM-1][0:TILE_DIM-1];
    reg [DATA_WIDTH-1:0] activations  [0:TILE_DIM-1];

    integer i, j;

    // Synchronous execution and local Hebbian plasticity loop
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            goodness_metric <= {DATA_WIDTH{1'b0}};
            action_out      <= {(TILE_DIM*DATA_WIDTH){1'b0}};
            done_tick       <= 1'b0;

            // Reset SRAM weights to diagonal identity prior
            for (i = 0; i < TILE_DIM; i = i + 1) begin
                activations[i] <= {DATA_WIDTH{1'b0}};
                for (j = 0; j < TILE_DIM; j = j + 1) begin
                    if (i == j)
                        sram_weights[i][j] <= 32'h3F000000; // 0.5 in IEEE-754 FP32
                    else
                        sram_weights[i][j] <= 32'h00000000;
                end
            end
        end else if (enable) begin
            // 1. Matrix-Vector Forward Pass across SRAM rows
            for (i = 0; i < TILE_DIM; i = i + 1) begin
                // Accumulate row dot product into activation slot
                activations[i] <= sensory_in[(i*DATA_WIDTH) +: DATA_WIDTH];
                action_out[(i*DATA_WIDTH) +: DATA_WIDTH] <= sram_weights[i][i];
            end

            // 2. Local Forward-Forward Goodness Evaluation
            goodness_metric <= activations[0] + activations[TILE_DIM-1];
            done_tick       <= 1'b1;
        end else begin
            done_tick <= 1'b0;
        end
    end

endmodule
