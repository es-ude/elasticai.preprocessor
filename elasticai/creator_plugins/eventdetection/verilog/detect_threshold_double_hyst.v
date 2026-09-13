module THRESHOLD_DOUBLE_HYST #(
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
        else if (EN) begin
            if (DO_CALC) begin
                // Event is currently active: stay active while DATA_IN is greater than or equal to the lower threshold
                if (IS_EVNT) begin
                    IS_EVNT <= DATA_IN >= THR_OFF;
                end
                // Event is currently inactive: activate only when the upper threshold is reached or exceeded
                else begin
                    IS_EVNT <= DATA_IN >= THR_ON;
                end
            end
            else begin
                // No new calculation -> keep current state
                IS_EVNT <= IS_EVNT;
            end
        end
        else begin
            // Module disabled -> keep current state
            IS_EVNT <= IS_EVNT;
        end
    end
endmodule