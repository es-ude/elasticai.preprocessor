//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     26.04.2023 18:22:50
// Copied on: 	    §{date_copy_created}
// Module Name:     Non-Recursive Polyphase Decimation Filter
// Target Devices:  FPGA
// Tool Versions:   1v0
// Processing:      Data applied on posedge clk, Gain = 2 ** POLY_ORDER
// Dependencies:    None
//
// State: 	        Works!
// Improvements:    None
// Parameters:      BITWIDTH --> Bitwidth of input data
//                  POLY_ORDER -->  Order of polyphase decimation filter
//////////////////////////////////////////////////////////////////////////////////


module FILTER_POLYDEC_FPGA#(
    parameter BITWIDTH = 12,
    parameter POLY_ORDER = 1
)(
    input wire CLK_SYS,
    input wire CLK_HGH,
	output reg CLK_LOW,
    input wire RSTN,
    input wire EN,
    input wire [BITWIDTH-'d1:0] DATA_IN,
    output reg [BITWIDTH+POLY_ORDER-'d1:0] DATA_OUT
);

	reg shift_clk_high;
	reg [BITWIDTH-'d1:0] din_dly [POLY_ORDER-'d1:0];
	wire do_low, do_hgh;

	assign do_low = EN && do_hgh && !CLK_LOW;
	assign do_hgh = EN && !shift_clk_high  && CLK_HGH;

	// --- Control sequence
	integer i0;
	always@(posedge CLK_SYS) begin
		if(~RSTN) begin
			 shift_clk_high <= 1'd0;
			 CLK_LOW <= 1'd0;
			 for (i0 = 'd0; i0 < POLY_ORDER; i0 = i0 + 'd1) begin
			    din_dly[i0] <= 'd0;
			 end
			 DATA_OUT <= 'd0;
		end else begin
		    // Clock Shifting
	        shift_clk_high <= (EN) ? CLK_HGH : 1'd0;

			// Clock Generation
			CLK_LOW <= (do_hgh) ? ~CLK_LOW : CLK_LOW;
            din_dly[0] <= (do_hgh && CLK_LOW) ? DATA_IN : din_dly[0];

			case(POLY_ORDER)
			   2'd1: begin
			       DATA_OUT <= (do_hgh && !CLK_LOW) ? {1'd0, DATA_IN} + {1'd0, din_dly[0]} : DATA_OUT;
               end
               2'd2: begin
                   DATA_OUT <= (do_hgh && !CLK_LOW) ? {2'b0, DATA_IN} + {1'b0, din_dly[0], 1'b0} + {2'b0, din_dly[1]} : DATA_OUT;
                   din_dly[1] <= (do_hgh && CLK_LOW) ? din_dly[0] : din_dly[1];
               end
			endcase
		end
	end

endmodule
