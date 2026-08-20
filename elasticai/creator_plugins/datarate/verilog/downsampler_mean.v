
module DOWNSAMPLER_MEAN #(
    parameter integer DATA_WIDTH = 16,             
    parameter integer DSR        = 8,              
    parameter integer ACC_WIDTH  = DATA_WIDTH + 8 
)(
    input  wire                          clk,
    input  wire                          rst_n,     
    input  wire                          in_valid,  
    input  wire signed [DATA_WIDTH-1:0]  din,       // Eingangssample 

    output reg                           out_valid, // 1 Takt lang high, wenn dout gültig ist
    output reg  signed [DATA_WIDTH-1:0]  dout        // Mittelwert der letzten DSR Samples
);
    localparam integer CNT_WIDTH = (DSR <= 1) ? 1 : $clog2(DSR);
    localparam          IS_POW2   = (DSR != 0) && ((DSR & (DSR - 1)) == 0);
    
    reg [CNT_WIDTH-1:0]        sample_cnt;
    reg signed [ACC_WIDTH-1:0] sum_reg;

    wire signed [ACC_WIDTH-1:0] sum_with_new_sample = sum_reg + din;
    wire                        group_complete      = in_valid && (sample_cnt == DSR - 1);
    wire signed [ACC_WIDTH-1:0] mean_value;


// Mittelwert: Bitshift bei Zweierpotenz-DSR, sonst echte Division.
    generate
        if (IS_POW2) begin : g_shift
            assign mean_value = sum_with_new_sample >>> $clog2(DSR);
        end else begin : g_div
            assign mean_value = sum_with_new_sample / DSR;
        end
    endgenerate

// Zähler, wird am Ende zurückgesetzt
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            sample_cnt <= {CNT_WIDTH{1'b0}};
        else if (in_valid)
            sample_cnt <= group_complete ? {CNT_WIDTH{1'b0}} : sample_cnt + 1'b1;
    end

// Akkumulator
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            sum_reg <= {ACC_WIDTH{1'b0}};
        else if (in_valid)
            sum_reg <= group_complete ? {ACC_WIDTH{1'b0}} : sum_with_new_sample;
    end

// Output: Mittelwert + Valid-Puls (1 Takt)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dout      <= {DATA_WIDTH{1'b0}};
            out_valid <= 1'b0;
        end else begin
            out_valid <= group_complete;
            if (group_complete)
                dout <= mean_value;
        end
    end

endmodule