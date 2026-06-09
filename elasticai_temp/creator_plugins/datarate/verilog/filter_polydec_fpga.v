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
    parameter BITWIDTH = 5'd12,
    parameter POLY_ORDER = 2'd1
)(
    input wire CLK_SYS,
    input wire CLK_HGH,
	output reg CLK_LOW,
    input wire RSTN,
    input wire EN,
    input wire [BITWIDTH-'d1:0] DATA_IN,
    output reg [BITWIDTH+POLY_ORDER-'d1:0] DATA_OUT
);
    localparam POLY_ORDER_EFF = (POLY_ORDER < 'd2) ? 'd0 : POLY_ORDER-'d2;

	reg shift_clk_high, shift_clk_low;
	reg [BITWIDTH-'d1:0] din_dly_hgh;
	reg [BITWIDTH-'d1:0] din_dly_low [POLY_ORDER_EFF:0];
	wire do_low, do_hgh;
	
	assign do_low = EN && !shift_clk_low   && CLK_LOW;
	assign do_hgh = EN && !shift_clk_high  && CLK_HGH;

	// --- Control sequence
	integer i0;
	always@(posedge CLK_SYS or negedge RSTN) begin
		if(~RSTN) begin
			 shift_clk_high <= 1'd0;
			 shift_clk_low <= 1'd0;
			 CLK_LOW <= 1'd0;
			 din_dly_hgh <= 'd0;
			 for(i0 = 'd0; i0 <= POLY_ORDER_EFF; i0 = i0 + 'd1) begin
			    din_dly_low[i0] <= 'd0;
			 end
			 DATA_OUT <= 'd0;
		end else begin
		    // Clock Shifting
	        shift_clk_high <= (EN) ? CLK_HGH : 1'd0;
			shift_clk_low <= (EN) ? CLK_LOW : 1'd0;
			// Clock Generation
			CLK_LOW <= (POLY_ORDER == 2'd0) ? CLK_HGH : ((do_hgh) ? ~CLK_LOW : CLK_LOW);
			// Data ProcessingSampling the data input on high sampling rate and Data calculating on low level
			din_dly_hgh <= (do_hgh) ? DATA_IN : din_dly_hgh;
			case(POLY_ORDER)
			   2'd1: begin
			       DATA_OUT <= (do_low) ? DATA_IN + din_dly_hgh : DATA_OUT;
               end
               2'd2: begin
                   DATA_OUT <= (do_low) ? DATA_IN + {din_dly_hgh, 1'd0} + din_dly_low[0] : DATA_OUT;
                   din_dly_low[0] <= (do_low) ? DATA_IN : din_dly_low[0];
               end
               2'd3: begin
                   DATA_OUT <= (do_low) ? DATA_IN + (din_dly_hgh + {din_dly_hgh, 1'd0}) + (din_dly_low[0] + {din_dly_low[0], 1'd0}) + din_dly_low[1] : DATA_OUT;
                   din_dly_low[0] <= (do_low) ? DATA_IN : din_dly_low[0];
                   din_dly_low[1] <= (do_low) ? din_dly_hgh : din_dly_low[1];
               end
               default: begin
                   DATA_OUT <= DATA_IN;
               end
			endcase
		end
	end

endmodule
