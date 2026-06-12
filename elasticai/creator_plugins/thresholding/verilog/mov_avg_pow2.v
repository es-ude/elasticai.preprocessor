//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
// 
// Create Date: 	21.10.2024 12:38:44
// Copied on: 	    §{date_copy_created}
// Module Name:     FIR-based Moving Average Filter (Binary division)
// Target Devices:  ASIC / FPGA
// Tool Versions:   1v1
// Description:     Moving Average with N = {$length} @ fs = {$sampling_rate} Hz for signed input data
// Processing:      Data applied on posedge clk
// Dependencies:    LENGTH is only a logarithmic value (otherwise result is invalid)
//                  Internal operation with unsigned values and scaling weight has fraction width of bitwidth
//
// State:		    Works! (System Test done: 29.10.2024 on Arty A7-35T with 20% usage)
// Improvements:    None
// Parameters:      BITWIDTH --> Bitwidth of input data
//                  LENGTH --> Length of used taps (=FIR filter order)
//////////////////////////////////////////////////////////////////////////////////


module MOVING_AVERAGE_POW2#(
    parameter BITWIDTH = 6'd8,
    parameter LENGTH = 9'd4
)(
    input wire CLK_SYS,
    input wire RSTN,
    input wire EN,
    input wire DO_CALC,
    input wire [BITWIDTH-'d1:0] DATA_IN,
    output wire [BITWIDTH-'d1:0] DATA_OUT,
    output wire DVALID
);
    localparam UPPER_MASK = BITWIDTH+$clog2(LENGTH);
    // --- Control Signals
    reg [1:0] do_calc_dly;
    reg first_run_done;
    reg [$clog2(LENGTH)-'d1:0] cnt_pos;
    reg [BITWIDTH-'d1:0] taps_fir [LENGTH-'d1:0];
    reg [UPPER_MASK-'d1:0] pre_out;
    wire do_process;

    assign do_process = ~do_calc_dly[1] && do_calc_dly[0];
    assign DVALID = first_run_done && ~do_process;
    assign DATA_OUT = pre_out[(UPPER_MASK-'d1)-:BITWIDTH];

    // --- Performing computation
    integer i0;
    always@(posedge CLK_SYS) begin
        if(~(RSTN && EN)) begin
            do_calc_dly <= 2'd0;
            cnt_pos <= 'd0;
            for(i0 = 0; i0 < LENGTH; i0 = i0 + 'd1) begin
                taps_fir[i0] = 'd0;
            end
            pre_out <= 'd0;
            first_run_done <= 1'd0;
        end else begin
            do_calc_dly <= {do_calc_dly[0], DO_CALC};
            if(do_process) begin
                taps_fir[cnt_pos] <= DATA_IN;
                pre_out <= pre_out - taps_fir[cnt_pos] + DATA_IN;
                first_run_done <= 1'd1;
                cnt_pos <= (cnt_pos == 'd0) ? LENGTH -'d1 : cnt_pos - 'd1;
            end else begin
                taps_fir[cnt_pos] <= taps_fir[cnt_pos];
                pre_out <= pre_out;
                cnt_pos <= cnt_pos;
                first_run_done <= first_run_done;
            end
        end
    end
endmodule
