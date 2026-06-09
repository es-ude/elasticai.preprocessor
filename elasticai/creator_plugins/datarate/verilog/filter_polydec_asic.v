//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     23.07.2025 13:12:11
// Copied on: 	    §{date_copy_created}
// Module Name:     Non-Recursive Polyphase Decimation Filter
// Target Devices:  ASIC
// Tool Versions:   1v0
// Processing:      Data applied on posedge clk, Gain = 2 ** POLY_ORDER
// Dependencies:    None
//
// State: 	        Works!
// Improvements:    None
// Parameters:      BITWIDTH --> Bitwidth of input data
//                  POLY_ORDER -->  Order of polyphase decimation filter
//////////////////////////////////////////////////////////////////////////////////


module FILTER_POLYDEC_ASIC#(
    parameter BITWIDTH = 5'd12,
    parameter POLY_ORDER = 2'd1
)(
    input wire CLK_HGH,
	output wire CLK_LOW,
    input wire RSTN,
    input wire EN,
    input wire [BITWIDTH-'d1:0] DATA_IN,
    output reg [BITWIDTH+POLY_ORDER-'d1:0] DATA_OUT
);
    localparam POLY_ORDER_EFF = (POLY_ORDER < 'd2) ? 'd0 : POLY_ORDER-'d2;

    reg clk_half;
	reg [BITWIDTH-'d1:0] din_dly_hgh;
	reg [BITWIDTH-'d1:0] din_dly_low [POLY_ORDER_EFF:0];
	assign CLK_LOW = (POLY_ORDER == 2'd0) ? CLK_HGH : clk_half;

	// --- Control sequence
	integer i0;
	always@(posedge CLK_HGH or negedge RSTN) begin
		if(~RSTN) begin
		     clk_half <= 1'd0;
			 din_dly_hgh <= 'd0;
			 for(i0 = 'd0; i0 <= POLY_ORDER_EFF; i0 = i0 + 'd1) begin
			    din_dly_low[i0] <= 'd0;
			 end
			 DATA_OUT <= 'd0;
		end else begin
			// Clock Generation
			clk_half <= ~clk_half;
			
			// Data ProcessingSampling the data input on high sampling rate and Data calculating on low level
			din_dly_hgh <= DATA_IN;
			case(POLY_ORDER)
			   2'd1: begin
			       DATA_OUT <= DATA_IN + din_dly_hgh;
               end
               2'd2: begin
                   DATA_OUT <= DATA_IN + {din_dly_hgh, 1'd0} + din_dly_low[0];
                   din_dly_low[0] <= DATA_IN;
               end
               2'd3: begin
                   DATA_OUT <= DATA_IN + (din_dly_hgh + {din_dly_hgh, 1'd0}) + (din_dly_low[0] + {din_dly_low[0], 1'd0}) + din_dly_low[1];
                   din_dly_low[0] <= DATA_IN;
                   din_dly_low[1] <= din_dly_hgh;
               end
               default: begin
                   DATA_OUT <= DATA_IN;
               end
			endcase
		end
	end

endmodule
