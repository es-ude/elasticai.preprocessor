module SUBSAMPLER #(
    parameter N          = 4,   // Jeden N-ten Wert durchlassen
    parameter DATA_WIDTH = 8   
)(
    input  wire                     clk,
    input  wire                     rst_n,     
    input  wire                     in_valid,  
    input  wire [DATA_WIDTH-1:0]    in_data,

    output reg                      out_valid, 
    output reg  [DATA_WIDTH-1:0]    out_data
);

    reg [$clog2(N)-1:0] counter;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            counter   <= 0;
            out_valid <= 1'b0;
            out_data  <= {DATA_WIDTH{1'b0}};
        end else begin
            out_valid <= 1'b0; // Standard: kein gültiger Output in diesem Takt

            if (in_valid) begin
                if (counter == N-1) begin
                    counter   <= 0;
                    out_data  <= in_data;
                    out_valid <= 1'b1;   // jetzt kommt der N-te Wert raus
                end else begin
                    counter <= counter + 1'b1;
                end
            end
        end
    end

endmodule