//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     08.06.2022 18:22:50
// Copied on: 	    §{date_copy_created}
// Module Name:     Cascaded Integrator Comb Filter for Data Rate Decimation of input streaming data
// Target Devices:  ASIC / FPGA
// Tool Versions:   1v0
// Processing:      Data applied on posedge clk
// Dependencies:    (DEC < 128), Delta Bitwidth Gain = N_DEC * log2(N_DEC)
//
// State: 	        Works!
// Improvements:    None
// Parameters:      BITWIDTH --> Bitwidth of input data
//                  DEC_RATE --> Decimation rate
//                  N_DEC --> Order of the integrator stage
// Information:     https://www.dsprelated.com/showarticle/1337.php
//////////////////////////////////////////////////////////////////////////////////


module FILTER_CIC#(
    parameter BITWIDTH = 5'd8,
    parameter DEC_RATE = 8'd128,
    parameter N_DEC = 4'd2
 )(
    input wire               				CLK_SYS,
    input wire                              CLK_SMP,
    input wire               				RSTN,
    input wire                              EN,
    input wire [BITWIDTH-'d1:0]                         DATA_IN,
    output wire [BITWIDTH+N_DEC*$clog2(DEC_RATE)-'d2:0] DATA_OUT,
    output wire 			 				DEC_CLK
);

    localparam BIT_OVR = N_DEC*$clog2(DEC_RATE)-'d1;

    // Integrator stage registers
    reg [BIT_OVR+BITWIDTH-'d1:0] dhigh, dout;
    reg [BIT_OVR+BITWIDTH-'d1:0] dlow [N_DEC-'d1:0];

    // Control signals
    reg shift_clk_hgh;
    reg [$clog2(DEC_RATE):0] count;
    reg v_comb;
    wire do_dec;

    assign do_dec = CLK_SMP && !shift_clk_hgh;
    assign DEC_CLK = v_comb;
    assign DATA_OUT = dout;

    integer i0;
    always@(posedge CLK_SYS) begin
        if(!EN || !RSTN) begin
            count <= 'd0;
            shift_clk_hgh <= 'd0;
            dhigh <= 'd0;
            for (i0 = 0; i0 < N_DEC; i0 = i0 + 'd1) begin
                dlow[i0] <= 'd0;
            end
            dout <= 'd0;
            v_comb <= 1'd0;
        end else begin
            shift_clk_hgh <= CLK_SMP;
            // Decimation decision
            if (count == DEC_RATE) begin
                count <= 'd0;
                v_comb <= 1'b1;
            end else begin
                count <= count + ((do_dec) ? 8'd1 : 8'd0);
                v_comb <= 1'b0;
            end

            // Integrator section running at sampling clock
            if(do_dec) begin
                dhigh <= dhigh + DATA_IN;
            end else begin
                dhigh <= dhigh;
            end

            // Comb section running at output rate
            dlow[0] <= (v_comb) ? dhigh : dlow[0];
            for (i0 = 1; i0 < N_DEC; i0 = i0 + 'd1) begin
                dlow[i0] <= (v_comb) ? dlow[i0-1] : dlow[i0];
            end
            dout <= (v_comb) ? (dhigh - dlow[N_DEC-'d1]) : dout;
        end
    end
endmodule
