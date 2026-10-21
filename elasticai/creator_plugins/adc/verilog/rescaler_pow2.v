//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        ND, AE
//
// Create Date:     20.10.2026, 18:06:32
// Copied on: 	    §{date_copy_created}
// Module Name:     Data Rescaler Function for Time Series Analysis (using bitshifting)
// Target Devices:  FPGA
// Tool Versions:   1v0
// Processing:      Data applied on posedge CLK_SYS
// Dependencies:    None
//
// State: 	        Not tested!
// Improvements:    None
// Parameters:      BITWIDTH_IN --> Bitwidth of input data
//////////////////////////////////////////////////////////////////////////////////


module RESCALER_POW2 #(
    parameter integer BITWIDTH_IN = 8,
    parameter integer IS_SIGNED_IN = 0,
    parameter integer BITWIDTH_OUT = 8,
    parameter integer IS_SIGNED_OUT = 0,
    parameter integer NUM_SAMPLES = 5
)(
    input wire                                  CLK_SYS,
    input wire                                  RSTN,
    input wire                                  EN,
    input wire                                  IN_VALID,
    input wire [NUM_SAMPLES* BITWIDTH_IN-1:0]   DATA_IN,
    output reg [NUM_SAMPLES* BITWIDTH_OUT-1:0]  DATA_OUT,
    output reg                                  DATA_RDY
);

    localparam integer ZERO_POINT = 0;
    localparam integer SHIFT = 0;
    localparam integer SYMMETRIC_RANGE = 0;

    localparam integer LSH   = (SHIFT < 0) ? -SHIFT : 0;
    localparam integer RSH_R = (SHIFT > 0) ? SHIFT : 0;
    localparam integer RSH   = (RSH_R > BITWIDTH_IN + 1) ? (BITWIDTH_IN + 1) : RSH_R;

    localparam integer W_IN  = BITWIDTH_IN + LSH + 3;
    localparam integer W_OUT = BITWIDTH_OUT + 2;
    localparam integer PROD_W = (W_IN > W_OUT) ? W_IN : W_OUT;

    localparam signed [PROD_W-1:0] ONE    = 1;
    localparam signed [PROD_W-1:0] ZP_V   = ZERO_POINT;
    localparam signed [PROD_W-1:0] HALF_V = (ONE <<< RSH) >>> 1;
    localparam signed [PROD_W-1:0] MAX_V  = IS_SIGNED_OUT ? ((ONE <<< (BITWIDTH_OUT - 1)) - 1) : ((ONE <<< BITWIDTH_OUT) - 1);
    localparam signed [PROD_W-1:0] MIN_V  = IS_SIGNED_OUT ? (SYMMETRIC_RANGE ? -MAX_V : -(ONE <<< (BITWIDTH_OUT - 1))) : 0;

    function [BITWIDTH_OUT-1:0] rescale;
        input [BITWIDTH_IN-1:0] raw;
        reg signed [PROD_W-1:0] x;
        reg signed [PROD_W-1:0] q;
        begin
            if (IS_SIGNED_IN != 0)
                x = $signed(raw);
            else
                x = $signed({{(PROD_W - BITWIDTH_IN){1'b0}}, raw});

            if (SHIFT < 0)
                q = x <<< LSH;
            else
                q = (x + HALF_V) >>> RSH;
            q = q + ZP_V;

            if (q > MAX_V)      q = MAX_V;
            else if (q < MIN_V) q = MIN_V;

            rescale = q[BITWIDTH_OUT-1:0];
        end
    endfunction

    reg [NUM_SAMPLES*BITWIDTH_IN-1:0] data_q;
    reg [$clog2(NUM_SAMPLES):0]       counter;
    reg                               in_valid_q;
    reg                               busy;

    always @(posedge CLK_SYS or negedge RSTN) begin
        if (!RSTN) begin
            data_q     <= {(NUM_SAMPLES * BITWIDTH_IN){1'b0}};
            in_valid_q <= 1'b0;
            busy       <= 1'b0;
            DATA_OUT   <= {(NUM_SAMPLES * BITWIDTH_OUT){1'b0}};
            DATA_RDY   <= 1'b0;
            counter    <= 'd0;
        end else begin
            in_valid_q <= IN_VALID;
            if (EN && IN_VALID && !in_valid_q) begin
                data_q   <= DATA_IN;
                busy     <= 1'b1;
                DATA_RDY <= 1'b0;
                DATA_OUT <= DATA_OUT;
                counter  <= 'd0;
            end else if (busy) begin
                DATA_OUT[counter*BITWIDTH_OUT +: BITWIDTH_OUT] <= rescale(data_q[counter*BITWIDTH_IN +: BITWIDTH_IN]);
                counter  <= counter + 'd1;
                busy     <= (counter == NUM_SAMPLES-1) ? 1'b0 : 1'b1;
                DATA_RDY <= (counter == NUM_SAMPLES-1) ? 1'b1 : 1'b0;
            end else begin
                DATA_OUT <= DATA_OUT;
                counter  <= counter;
                busy     <= busy;
                DATA_RDY <= DATA_RDY;
            end
        end
    end
endmodule
