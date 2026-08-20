//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        ND, AE
//
// Create Date:     10.08.2026, 17:54:12
// Copied on: 	    §{date_copy_created}
// Module Name:     Simple Datarate Subsampler
// Target Devices:  FPGA
// Tool Versions:   1v0
// Processing:      Data applied on posedge CLK_SYS
// Dependencies:    None
//
// State: 	        Not tested!
// Improvements:    None
// Parameters:      BITWIDTH --> Bitwidth of input data
//                  DEC_RATE --> Decimation rate
//////////////////////////////////////////////////////////////////////////////////


module DOWNSAMPLER_MEAN #(
    parameter integer BITWIDTH = 16,
    parameter integer DEC_RATE = 8
)(
    input wire                          CLK_SYS,
    input wire                          RSTN,
    input wire                          EN,
    input wire                          IN_VALID,
    input wire signed [BITWIDTH-1:0]    DATA_IN,
    output reg                          DATA_RDY,
    output reg  signed [BITWIDTH-1:0]   DATA_OUT
);

    localparam integer ACC_WIDTH = BITWIDTH + $clog2(DEC_RATE);
    localparam integer IS_POW2 = (DEC_RATE != 0) && ((DEC_RATE & (DEC_RATE - 1)) == 0);

    reg in_valid_dly;
    reg [$clog2(DEC_RATE):0] sample_cnt;
    reg signed [ACC_WIDTH-1:0] sum_reg;

    always @(posedge CLK_SYS) begin
        if (!RSTN) begin
            in_valid_dly <= 1'd0;
            sample_cnt <= DEC_RATE - 'd1;
            sum_reg    <= 'd0;
            DATA_RDY  <= 1'b0;
            DATA_OUT  <= 'd0;
        end else begin
            in_valid_dly <= IN_VALID;
            if (IN_VALID && !in_valid_dly && EN) begin
                sample_cnt <= (sample_cnt == DEC_RATE - 'd1) ? 'd0 : sample_cnt + 'd1;
                sum_reg    <= (sample_cnt == DEC_RATE - 'd1) ? {{$clog2(DEC_RATE){DATA_IN[BITWIDTH-1]}}, DATA_IN} : sum_reg + {{$clog2(DEC_RATE){DATA_IN[BITWIDTH-1]}}, DATA_IN};
                DATA_OUT <= (sample_cnt == DEC_RATE - 'd1) ? ((IS_POW2) ? (sum_reg >>> $clog2(DEC_RATE)) : (sum_reg / DEC_RATE)) : DATA_OUT;
                DATA_RDY <= (sample_cnt == DEC_RATE - 'd1);
            end else begin
                sample_cnt <= sample_cnt;
                sum_reg <= sum_reg;
                DATA_OUT <= DATA_OUT;
                DATA_RDY <= DATA_RDY;
            end
        end
    end
endmodule
