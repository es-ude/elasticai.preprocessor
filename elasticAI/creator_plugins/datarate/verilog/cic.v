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
    output wire [BITWIDTH+N_DEC*$clog2(DEC_RATE)-'d1:0] DATA_OUT,
    output wire 			 				DEC_CLK
);

    localparam BIT_OVR = N_DEC*$clog2(DEC_RATE);

    // Integrator stage registers: N_DEC kaskadierte Integratoren statt nur einem
    reg [BIT_OVR+BITWIDTH-'d1:0] dhigh [N_DEC-'d1:0];
    reg [BIT_OVR+BITWIDTH-'d1:0] dlow [N_DEC-'d1:0];
    reg [BIT_OVR+BITWIDTH-'d1:0] dout;

    // Kombinatorische Integrator-Kaskade: alle N_DEC Stufen im selben Sample
    // (analog zu "for i in range(num_stages): z = intes[i].update(z)")
    wire [BIT_OVR+BITWIDTH-'d1:0] integ_chain [N_DEC:0];
    assign integ_chain[0] = DATA_IN;
    genvar gi;
    generate
        for (gi = 0; gi < N_DEC; gi = gi + 1) begin: integ_stage
            assign integ_chain[gi+1] = dhigh[gi] + integ_chain[gi];
        end
    endgenerate

    // Kombinatorische Comb-Kaskade: alle N_DEC Stufen im selben Takt
    // (analog zu "for c in range(num_stages): z = combs[c].update(z)")
    wire [BIT_OVR+BITWIDTH-'d1:0] comb_chain [N_DEC:0];
    assign comb_chain[0] = dhigh[N_DEC-'d1];
    genvar gc;
    generate
        for (gc = 0; gc < N_DEC; gc = gc + 1) begin: comb_stage
            assign comb_chain[gc+1] = comb_chain[gc] - dlow[gc];
        end
    endgenerate

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
            for (i0 = 0; i0 < N_DEC; i0 = i0 + 'd1) begin
                dhigh[i0] <= 'd0;
                dlow[i0] <= 'd0;
            end
            dout <= 'd0;
            v_comb <= 1'd0;
        end else begin
            shift_clk_hgh <= CLK_SMP;

            // Decimation decision: Trigger bei sample_index % DEC_RATE == 0
            // (analog zu "if (s % dsr) == 0" in do_cic, inkl. dem allerersten Sample)
            if (do_dec) begin
                v_comb <= (count == 'd0) ? 1'b1 : 1'b0;
                count <= (count == DEC_RATE-'d1) ? 'd0 : count + 8'd1;
            end else begin
                v_comb <= 1'b0;
            end

            // Integrator section running at sampling clock (N_DEC Stufen kaskadiert)
            if(do_dec) begin
                for (i0 = 0; i0 < N_DEC; i0 = i0 + 1) begin
                    dhigh[i0] <= integ_chain[i0+1];
                end
            end

            // Comb section running at output rate (N_DEC Stufen kaskadiert)
            if (v_comb) begin
                for (i0 = 0; i0 < N_DEC; i0 = i0 + 1) begin
                    dlow[i0] <= comb_chain[i0];
                end
            end
            dout <= (v_comb) ? comb_chain[N_DEC] : dout;
        end
    end
endmodule
