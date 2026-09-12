module UNSIGNED_THRESHOLD#(
    parameter integer BITWIDTH = 8
)(
    input wire CLK_SYS,
    input wire RSTN,
    input wire EN,
    input wire DO_CALC,
    input wire [BITWIDTH-1:0] DATA_IN,
    input wire [BITWIDTH-1:0] THR,
    output reg IS_EVNT,
    output wire DVALID
);

    assign DVALID = DO_CALC;

    always @(posedge CLK_SYS) begin
        if (!RSTN) begin
            IS_EVNT <= 1'b0;
        end
        else if (EN) begin
            if (DO_CALC) begin
                IS_EVNT <= DATA_IN >= THR;
            end
            else begin
                // Kein neuer Berechnungsvorgang
                IS_EVNT <= IS_EVNT;
            end
        end
    end

endmodule