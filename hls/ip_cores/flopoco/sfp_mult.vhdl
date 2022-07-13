--------------------------------------------------------------------------------
--                           DSPBlock_5x5_F300_uid9
-- VHDL generated for Virtex6 @ 300MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: 
--------------------------------------------------------------------------------
-- Pipeline depth: 0 cycles
-- Clock period (ns): 3.33333
-- Target frequency (MHz): 300
-- Input signals: X Y
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library std;
use std.textio.all;
library work;

entity DSPBlock_5x5_F300_uid9 is
    port (clk : in std_logic;
          X : in  std_logic_vector(4 downto 0);
          Y : in  std_logic_vector(4 downto 0);
          R : out  std_logic_vector(9 downto 0)   );
end entity;

architecture arch of DSPBlock_5x5_F300_uid9 is
signal Mint :  std_logic_vector(9 downto 0);
signal M :  std_logic_vector(9 downto 0);
signal Rtmp :  std_logic_vector(9 downto 0);
begin
   Mint <= std_logic_vector(unsigned(X) * unsigned(Y)); -- multiplier
   M <= Mint(9 downto 0);
   Rtmp <= M;
   R <= Rtmp;
end architecture;

--------------------------------------------------------------------------------
--                          IntMultiplier_F300_uid5
-- VHDL generated for Virtex6 @ 300MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Martin Kumm, Florent de Dinechin, Kinga Illyes, Bogdan Popa, Bogdan Pasca, 2012
--------------------------------------------------------------------------------
-- Pipeline depth: 0 cycles
-- Clock period (ns): 3.33333
-- Target frequency (MHz): 300
-- Input signals: X Y
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library std;
use std.textio.all;
library work;

entity IntMultiplier_F300_uid5 is
    port (clk : in std_logic;
          X : in  std_logic_vector(4 downto 0);
          Y : in  std_logic_vector(4 downto 0);
          R : out  std_logic_vector(9 downto 0)   );
end entity;

architecture arch of IntMultiplier_F300_uid5 is
   component DSPBlock_5x5_F300_uid9 is
      port ( clk : in std_logic;
             X : in  std_logic_vector(4 downto 0);
             Y : in  std_logic_vector(4 downto 0);
             R : out  std_logic_vector(9 downto 0)   );
   end component;

signal XX_m6 :  std_logic_vector(4 downto 0);
signal YY_m6 :  std_logic_vector(4 downto 0);
signal tile_0_X :  std_logic_vector(4 downto 0);
signal tile_0_Y :  std_logic_vector(4 downto 0);
signal tile_0_output :  std_logic_vector(9 downto 0);
signal tile_0_filtered_output :  unsigned(9-0 downto 0);
signal bh7_w0_0 :  std_logic;
signal bh7_w1_0 :  std_logic;
signal bh7_w2_0 :  std_logic;
signal bh7_w3_0 :  std_logic;
signal bh7_w4_0 :  std_logic;
signal bh7_w5_0 :  std_logic;
signal bh7_w6_0 :  std_logic;
signal bh7_w7_0 :  std_logic;
signal bh7_w8_0 :  std_logic;
signal bh7_w9_0 :  std_logic;
signal tmp_bitheapResult_bh7_9 :  std_logic_vector(9 downto 0);
signal bitheapResult_bh7 :  std_logic_vector(9 downto 0);
begin
   XX_m6 <= X ;
   YY_m6 <= Y ;
   tile_0_X <= X(4 downto 0);
   tile_0_Y <= Y(4 downto 0);
   tile_0_mult: DSPBlock_5x5_F300_uid9
      port map ( clk  => clk,
                 X => tile_0_X,
                 Y => tile_0_Y,
                 R => tile_0_output);

   tile_0_filtered_output <= unsigned(tile_0_output(9 downto 0));
   bh7_w0_0 <= tile_0_filtered_output(0);
   bh7_w1_0 <= tile_0_filtered_output(1);
   bh7_w2_0 <= tile_0_filtered_output(2);
   bh7_w3_0 <= tile_0_filtered_output(3);
   bh7_w4_0 <= tile_0_filtered_output(4);
   bh7_w5_0 <= tile_0_filtered_output(5);
   bh7_w6_0 <= tile_0_filtered_output(6);
   bh7_w7_0 <= tile_0_filtered_output(7);
   bh7_w8_0 <= tile_0_filtered_output(8);
   bh7_w9_0 <= tile_0_filtered_output(9);

   -- Adding the constant bits 
      -- All the constant bits are zero, nothing to add

   tmp_bitheapResult_bh7_9 <= bh7_w9_0 & bh7_w8_0 & bh7_w7_0 & bh7_w6_0 & bh7_w5_0 & bh7_w4_0 & bh7_w3_0 & bh7_w2_0 & bh7_w1_0 & bh7_w0_0;
   bitheapResult_bh7 <= tmp_bitheapResult_bh7_9;
   R <= bitheapResult_bh7(9 downto 0);
end architecture;

--------------------------------------------------------------------------------
--                           IntAdder_10_F300_uid13
-- VHDL generated for Virtex6 @ 300MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Bogdan Pasca, Florent de Dinechin (2008-2016)
--------------------------------------------------------------------------------
-- Pipeline depth: 1 cycles
-- Clock period (ns): 3.33333
-- Target frequency (MHz): 300
-- Input signals: X Y Cin
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity IntAdder_10_F300_uid13 is
    port (clk : in std_logic;
          X : in  std_logic_vector(9 downto 0);
          Y : in  std_logic_vector(9 downto 0);
          Cin : in  std_logic;
          R : out  std_logic_vector(9 downto 0)   );
end entity;

architecture arch of IntAdder_10_F300_uid13 is
signal Cin_1, Cin_1_d1 :  std_logic;
signal X_1, X_1_d1 :  std_logic_vector(10 downto 0);
signal Y_1, Y_1_d1 :  std_logic_vector(10 downto 0);
signal S_1 :  std_logic_vector(10 downto 0);
signal R_1 :  std_logic_vector(9 downto 0);
begin
   process(clk)
      begin
         if clk'event and clk = '1' then
            Cin_1_d1 <=  Cin_1;
            X_1_d1 <=  X_1;
            Y_1_d1 <=  Y_1;
         end if;
      end process;
   Cin_1 <= Cin;
   X_1 <= '0' & X(9 downto 0);
   Y_1 <= '0' & Y(9 downto 0);
   S_1 <= X_1_d1 + Y_1_d1 + Cin_1_d1;
   R_1 <= S_1(9 downto 0);
   R <= R_1 ;
end architecture;

--------------------------------------------------------------------------------
--                                    fmul
--                    (FPMult_4_4_4_4_4_4_uid2_F300_uid3)
-- VHDL generated for Virtex6 @ 300MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Bogdan Pasca, Florent de Dinechin 2008-2021
--------------------------------------------------------------------------------
-- Pipeline depth: 1 cycles
-- Clock period (ns): 3.33333
-- Target frequency (MHz): 300
-- Input signals: X Y
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity fmul is
    port (clk : in std_logic;
          X : in  std_logic_vector(4+4+2 downto 0);
          Y : in  std_logic_vector(4+4+2 downto 0);
          R : out  std_logic_vector(4+4+2 downto 0)   );
end entity;

architecture arch of fmul is
   component IntMultiplier_F300_uid5 is
      port ( clk : in std_logic;
             X : in  std_logic_vector(4 downto 0);
             Y : in  std_logic_vector(4 downto 0);
             R : out  std_logic_vector(9 downto 0)   );
   end component;

   component IntAdder_10_F300_uid13 is
      port ( clk : in std_logic;
             X : in  std_logic_vector(9 downto 0);
             Y : in  std_logic_vector(9 downto 0);
             Cin : in  std_logic;
             R : out  std_logic_vector(9 downto 0)   );
   end component;

signal sign, sign_d1 :  std_logic;
signal expX :  std_logic_vector(3 downto 0);
signal expY :  std_logic_vector(3 downto 0);
signal expSumPreSub :  std_logic_vector(5 downto 0);
signal bias :  std_logic_vector(5 downto 0);
signal expSum :  std_logic_vector(5 downto 0);
signal sigX :  std_logic_vector(4 downto 0);
signal sigY :  std_logic_vector(4 downto 0);
signal sigProd :  std_logic_vector(9 downto 0);
signal excSel :  std_logic_vector(3 downto 0);
signal exc, exc_d1 :  std_logic_vector(1 downto 0);
signal norm :  std_logic;
signal expPostNorm :  std_logic_vector(5 downto 0);
signal sigProdExt :  std_logic_vector(9 downto 0);
signal expSig :  std_logic_vector(9 downto 0);
signal sticky :  std_logic;
signal guard :  std_logic;
signal round :  std_logic;
signal expSigPostRound :  std_logic_vector(9 downto 0);
signal excPostNorm :  std_logic_vector(1 downto 0);
signal finalExc :  std_logic_vector(1 downto 0);
begin
   process(clk)
      begin
         if clk'event and clk = '1' then
            sign_d1 <=  sign;
            exc_d1 <=  exc;
         end if;
      end process;
   sign <= X(8) xor Y(8);
   expX <= X(7 downto 4);
   expY <= Y(7 downto 4);
   expSumPreSub <= ("00" & expX) + ("00" & expY);
   bias <= CONV_STD_LOGIC_VECTOR(7,6);
   expSum <= expSumPreSub - bias;
   sigX <= "1" & X(3 downto 0);
   sigY <= "1" & Y(3 downto 0);
   SignificandMultiplication: IntMultiplier_F300_uid5
      port map ( clk  => clk,
                 X => sigX,
                 Y => sigY,
                 R => sigProd);
   excSel <= X(10 downto 9) & Y(10 downto 9);
   with excSel  select  
   exc <= "00" when  "0000" | "0001" | "0100", 
          "01" when "0101",
          "10" when "0110" | "1001" | "1010" ,
          "11" when others;
   norm <= sigProd(9);
   -- exponent update
   expPostNorm <= expSum + ("00000" & norm);
   -- significand normalization shift
   sigProdExt <= sigProd(8 downto 0) & "0" when norm='1' else
                         sigProd(7 downto 0) & "00";
   expSig <= expPostNorm & sigProdExt(9 downto 6);
   sticky <= sigProdExt(5);
   guard <= '0' when sigProdExt(4 downto 0)="00000" else '1';
   round <= sticky and ( (guard and not(sigProdExt(6))) or (sigProdExt(6) ))  ;
   RoundingAdder: IntAdder_10_F300_uid13
      port map ( clk  => clk,
                 Cin => round,
                 X => expSig,
                 Y => "0000000000",
                 R => expSigPostRound);
   with expSigPostRound(9 downto 8)  select 
   excPostNorm <=  "01"  when  "00",
                               "10"             when "01", 
                               "00"             when "11"|"10",
                               "11"             when others;
   with exc_d1  select  
   finalExc <= exc_d1 when  "11"|"10"|"00",
                       excPostNorm when others; 
   R <= finalExc & sign_d1 & expSigPostRound(7 downto 0);
end architecture;

