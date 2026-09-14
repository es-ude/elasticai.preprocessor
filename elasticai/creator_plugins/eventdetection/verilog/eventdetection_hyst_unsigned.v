module UNSIGNED_HYSTERESIS #(
    parameter integer BITWIDTH = 8
)(
    input wire CLK_SYS,
    input wire RSTN,
    input wire EN,
    input wire DO_CALC,
    input wire [BITWIDTH-1:0] DATA_IN,
    input wire [BITWIDTH-1:0] THR_ON,
    input wire [BITWIDTH-1:0] THR_OFF,

    output reg IS_EVNT,
    output wire DVALID
);

    assign DVALID = DO_CALC;

    always @(posedge CLK_SYS) begin
        if (!RSTN) begin
            IS_EVNT <= 1'b0;
        end
        else if (EN && DO_CALC) begin
            if (IS_EVNT)
                IS_EVNT <= DATA_IN >= THR_OFF;
            else
                IS_EVNT <= DATA_IN >= THR_ON;
        end
    end
endmodule