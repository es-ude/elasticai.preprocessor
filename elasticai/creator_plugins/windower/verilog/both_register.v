module BOTH_REGISTER#(
    parameter BITWIDTH = 6'd12,
    parameter SAMPLES = 6'd2
)(
    input wire CLK_SYS,
    input wire RSTN,
    input wire EN,
    input wire DO_SHIFT,
    input wire [BITWIDTH-'d1:0] DATA_IN,
    output wire [BITWIDTH-'d1:0] DATA_OUT0,
    output wire [BITWIDTH-'d1:0] DATA_OUT1,
    output wire [BITWIDTH* SAMPLES-'d1:0] DATA_BUF0,
    output wire [BITWIDTH* SAMPLES-'d1:0] DATA_BUF1,
    output wire DVALID
);

    wire [1:0] data_valid;
    assign DVALID = &data_valid;

    RING_BUFFER#(BITWIDTH, SAMPLES) DUT0(
        .CLK_SYS(CLK_SYS),
        .RSTN(RSTN),
        .EN(EN),
        .DO_SHIFT(DO_SHIFT),
        .DATA_IN(DATA_IN),
        .DATA_OUT(DATA_OUT0),
        .DATA_BUF(DATA_BUF0),
        .DVALID(data_valid[0])
    );

    SHIFT_REGISTER#(BITWIDTH, SAMPLES) DUT1(
        .CLK_SYS(CLK_SYS),
        .RSTN(RSTN),
        .EN(EN),
        .DO_SHIFT(DO_SHIFT),
        .DATA_IN(DATA_IN),
        .DATA_OUT(DATA_OUT1),
        .DATA_BUF(DATA_BUF1),
        .DVALID(data_valid[1])
    );
endmodule
