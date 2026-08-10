//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
// 
// Create Date: 	10.08.2026 12:38:44
// Copied on: 	    §{date_copy_created}
// Module Name:     FIR-based Moving Average Absolute Filter
// Target Devices:  ASIC / FPGA
// Tool Versions:   1v1
// Description:     Moving Average with N = {$length} for signed input data
// Processing:      Data applied on posedge clk
// Dependencies:    LENGTH is only a logarithmic value (otherwise result is invalid)
//                  Internal operation with unsigned values and scaling weight has fraction width of bitwidth
//
// State:		    Not tested!
// Improvements:    None
// Parameters:      BITWIDTH --> Bitwidth of input data
//                  LENGTH --> Length of window (=number of taps)
//////////////////////////////////////////////////////////////////////////////////


module MOVING_AVERAGE#(
    parameter integer BITWIDTH = 8,
    parameter integer LENGTH = 4
)(
    input wire CLK_SYS,
    input wire RSTN,
    input wire EN,
    input wire DO_CALC,
    input wire signed [BITWIDTH-'d1:0] DATA_IN,
    output wire signed [BITWIDTH-'d1:0] DATA_OUT,
    output wire DVALID
);
    // --- Control Signals
    reg do_calc_dly;
    reg first_run_done;
    reg [$clog2(LENGTH)-'d1:0] cnt_pos;
    reg signed [BITWIDTH-'d1:0] taps_fir [LENGTH-'d1:0];
    reg signed [BITWIDTH + $clog2(LENGTH)-'d1:0] pre_out;

    assign DVALID = first_run_done && do_calc_dly && !DO_CALC;
    assign DATA_OUT = pre_out[$clog2(LENGTH)+:BITWIDTH];

    // --- Performing computation
    integer i0;
    always@(posedge CLK_SYS) begin
        if(!RSTN) begin
            do_calc_dly <= 1'd0;
            for(i0 = 0; i0 < LENGTH; i0 = i0 + 'd1) begin
                taps_fir[i0] = 'd0;
            end
            pre_out <= 'd0;
            first_run_done <= 1'd0;
            cnt_pos <= 'd0;
        end else begin
            do_calc_dly <= DO_CALC;
            if(!do_calc_dly && DO_CALC && EN) begin
                taps_fir[cnt_pos] <= (DATA_IN[BITWIDTH-'d1]) ? ~DATA_IN : DATA_IN;
                pre_out <= pre_out - taps_fir[cnt_pos] + ((DATA_IN[BITWIDTH-'d1]) ? ~DATA_IN : DATA_IN);
                first_run_done <= 1'd1;
                cnt_pos <= (cnt_pos == 'd0) ? LENGTH -'d1 : cnt_pos - 'd1;
            end else begin
                taps_fir[cnt_pos] <= taps_fir[cnt_pos];
                pre_out <= pre_out;
                first_run_done <= first_run_done;
                cnt_pos <= cnt_pos;
            end
        end
    end
endmodule
