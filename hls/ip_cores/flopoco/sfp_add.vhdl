--------------------------------------------------------------------------------
--                     RightShifter_5_by_max_7_F800_uid4
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Bogdan Pasca, Florent de Dinechin (2008-2011)
--------------------------------------------------------------------------------
-- Pipeline depth: 1 cycles

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity RightShifter_5_by_max_7_F800_uid4 is
   port ( clk, rst : in std_logic;
          X : in  std_logic_vector(4 downto 0);
          S : in  std_logic_vector(2 downto 0);
          R : out  std_logic_vector(11 downto 0)   );
end entity;

architecture arch of RightShifter_5_by_max_7_F800_uid4 is
signal level0 :  std_logic_vector(4 downto 0);
signal ps, ps_d1 :  std_logic_vector(2 downto 0);
signal level1 :  std_logic_vector(5 downto 0);
signal level2, level2_d1 :  std_logic_vector(7 downto 0);
signal level3 :  std_logic_vector(11 downto 0);
begin
   process(clk)
      begin
         if clk'event and clk = '1' then
            ps_d1 <=  ps;
            level2_d1 <=  level2;
         end if;
      end process;
   level0<= X;
   ps<= S;
   level1<=  (0 downto 0 => '0') & level0 when ps(0) = '1' else    level0 & (0 downto 0 => '0');
   level2<=  (1 downto 0 => '0') & level1 when ps(1) = '1' else    level1 & (1 downto 0 => '0');
   ----------------Synchro barrier, entering cycle 1----------------
   level3<=  (3 downto 0 => '0') & level2_d1 when ps_d1(2) = '1' else    level2_d1 & (3 downto 0 => '0');
   R <= level3(11 downto 0);
end architecture;

--------------------------------------------------------------------------------
--                            IntAdder_8_f800_uid8
--                      (IntAdderClassical_8_F800_uid10)
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Bogdan Pasca, Florent de Dinechin (2008-2010)
--------------------------------------------------------------------------------
-- Pipeline depth: 0 cycles

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity IntAdder_8_f800_uid8 is
   port ( clk, rst : in std_logic;
          X : in  std_logic_vector(7 downto 0);
          Y : in  std_logic_vector(7 downto 0);
          Cin : in  std_logic;
          R : out  std_logic_vector(7 downto 0)   );
end entity;

architecture arch of IntAdder_8_f800_uid8 is
begin
   process(clk)
      begin
         if clk'event and clk = '1' then
         end if;
      end process;
   --Classical
    R <= X + Y + Cin;
end architecture;

--------------------------------------------------------------------------------
--                  LZCShifter_9_to_9_counting_16_F800_uid16
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Florent de Dinechin, Bogdan Pasca (2007)
--------------------------------------------------------------------------------
-- Pipeline depth: 3 cycles

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity LZCShifter_9_to_9_counting_16_F800_uid16 is
   port ( clk, rst : in std_logic;
          I : in  std_logic_vector(8 downto 0);
          Count : out  std_logic_vector(3 downto 0);
          O : out  std_logic_vector(8 downto 0)   );
end entity;

architecture arch of LZCShifter_9_to_9_counting_16_F800_uid16 is
signal level4, level4_d1 :  std_logic_vector(8 downto 0);
signal count3, count3_d1, count3_d2, count3_d3 :  std_logic;
signal level3 :  std_logic_vector(8 downto 0);
signal count2, count2_d1, count2_d2 :  std_logic;
signal level2, level2_d1 :  std_logic_vector(8 downto 0);
signal count1, count1_d1 :  std_logic;
signal level1, level1_d1 :  std_logic_vector(8 downto 0);
signal count0, count0_d1 :  std_logic;
signal level0 :  std_logic_vector(8 downto 0);
signal sCount :  std_logic_vector(3 downto 0);
begin
   process(clk)
      begin
         if clk'event and clk = '1' then
            level4_d1 <=  level4;
            count3_d1 <=  count3;
            count3_d2 <=  count3_d1;
            count3_d3 <=  count3_d2;
            count2_d1 <=  count2;
            count2_d2 <=  count2_d1;
            level2_d1 <=  level2;
            count1_d1 <=  count1;
            level1_d1 <=  level1;
            count0_d1 <=  count0;
         end if;
      end process;
   level4 <= I ;
   count3<= '1' when level4(8 downto 1) = (8 downto 1=>'0') else '0';
   ----------------Synchro barrier, entering cycle 1----------------
   level3<= level4_d1(8 downto 0) when count3_d1='0' else level4_d1(0 downto 0) & (7 downto 0 => '0');

   count2<= '1' when level3(8 downto 5) = (8 downto 5=>'0') else '0';
   level2<= level3(8 downto 0) when count2='0' else level3(4 downto 0) & (3 downto 0 => '0');

   ----------------Synchro barrier, entering cycle 2----------------
   count1<= '1' when level2_d1(8 downto 7) = (8 downto 7=>'0') else '0';
   level1<= level2_d1(8 downto 0) when count1='0' else level2_d1(6 downto 0) & (1 downto 0 => '0');

   count0<= '1' when level1(8 downto 8) = (8 downto 8=>'0') else '0';
   ----------------Synchro barrier, entering cycle 3----------------
   level0<= level1_d1(8 downto 0) when count0_d1='0' else level1_d1(7 downto 0) & (0 downto 0 => '0');

   O <= level0;
   sCount <= count3_d3 & count2_d2 & count1_d1 & count0_d1;
   Count <= sCount;
end architecture;

--------------------------------------------------------------------------------
--                           IntAdder_11_f800_uid20
--                     (IntAdderClassical_11_F800_uid22)
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Bogdan Pasca, Florent de Dinechin (2008-2010)
--------------------------------------------------------------------------------
-- Pipeline depth: 0 cycles

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity IntAdder_11_f800_uid20 is
   port ( clk, rst : in std_logic;
          X : in  std_logic_vector(10 downto 0);
          Y : in  std_logic_vector(10 downto 0);
          Cin : in  std_logic;
          R : out  std_logic_vector(10 downto 0)   );
end entity;

architecture arch of IntAdder_11_f800_uid20 is
begin
   process(clk)
      begin
         if clk'event and clk = '1' then
         end if;
      end process;
   --Classical
    R <= X + Y + Cin;
end architecture;

--------------------------------------------------------------------------------
--                                  forward
--                           (FPAdd_4_4_F800_uid2)
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Bogdan Pasca, Florent de Dinechin (2010)
--------------------------------------------------------------------------------
-- Pipeline depth: 7 cycles

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity forward is
   port ( clk, rst : in std_logic;
          X : in  std_logic_vector(4+4+2 downto 0);
          Y : in  std_logic_vector(4+4+2 downto 0);
          R : out  std_logic_vector(4+4+2 downto 0)   );
end entity;

architecture arch of forward is
   component RightShifter_5_by_max_7_F800_uid4 is
      port ( clk, rst : in std_logic;
             X : in  std_logic_vector(4 downto 0);
             S : in  std_logic_vector(2 downto 0);
             R : out  std_logic_vector(11 downto 0)   );
   end component;

   component IntAdder_8_f800_uid8 is
      port ( clk, rst : in std_logic;
             X : in  std_logic_vector(7 downto 0);
             Y : in  std_logic_vector(7 downto 0);
             Cin : in  std_logic;
             R : out  std_logic_vector(7 downto 0)   );
   end component;

   component LZCShifter_9_to_9_counting_16_F800_uid16 is
      port ( clk, rst : in std_logic;
             I : in  std_logic_vector(8 downto 0);
             Count : out  std_logic_vector(3 downto 0);
             O : out  std_logic_vector(8 downto 0)   );
   end component;

   component IntAdder_11_f800_uid20 is
      port ( clk, rst : in std_logic;
             X : in  std_logic_vector(10 downto 0);
             Y : in  std_logic_vector(10 downto 0);
             Cin : in  std_logic;
             R : out  std_logic_vector(10 downto 0)   );
   end component;

signal excExpFracX :  std_logic_vector(9 downto 0);
signal excExpFracY :  std_logic_vector(9 downto 0);
signal eXmeY, eXmeY_d1 :  std_logic_vector(4 downto 0);
signal eYmeX, eYmeX_d1 :  std_logic_vector(4 downto 0);
signal swap, swap_d1 :  std_logic;
signal newX, newX_d1, newX_d2 :  std_logic_vector(10 downto 0);
signal newY :  std_logic_vector(10 downto 0);
signal expX, expX_d1, expX_d2 :  std_logic_vector(3 downto 0);
signal excX :  std_logic_vector(1 downto 0);
signal excY :  std_logic_vector(1 downto 0);
signal signX, signX_d1 :  std_logic;
signal signY :  std_logic;
signal EffSub, EffSub_d1, EffSub_d2, EffSub_d3, EffSub_d4, EffSub_d5, EffSub_d6 :  std_logic;
signal sXsYExnXY, sXsYExnXY_d1 :  std_logic_vector(5 downto 0);
signal sdExnXY :  std_logic_vector(3 downto 0);
signal fracY :  std_logic_vector(4 downto 0);
signal excRt, excRt_d1, excRt_d2, excRt_d3, excRt_d4, excRt_d5, excRt_d6 :  std_logic_vector(1 downto 0);
signal signR, signR_d1, signR_d2, signR_d3, signR_d4, signR_d5 :  std_logic;
signal expDiff :  std_logic_vector(4 downto 0);
signal shiftedOut :  std_logic;
signal shiftVal :  std_logic_vector(2 downto 0);
signal shiftedFracY, shiftedFracY_d1 :  std_logic_vector(11 downto 0);
signal sticky :  std_logic;
signal fracYfar :  std_logic_vector(7 downto 0);
signal EffSubVector :  std_logic_vector(7 downto 0);
signal fracYfarXorOp :  std_logic_vector(7 downto 0);
signal fracXfar :  std_logic_vector(7 downto 0);
signal cInAddFar :  std_logic;
signal fracAddResult :  std_logic_vector(7 downto 0);
signal fracGRS :  std_logic_vector(8 downto 0);
signal extendedExpInc, extendedExpInc_d1, extendedExpInc_d2, extendedExpInc_d3 :  std_logic_vector(5 downto 0);
signal nZerosNew :  std_logic_vector(3 downto 0);
signal shiftedFrac :  std_logic_vector(8 downto 0);
signal updatedExp :  std_logic_vector(5 downto 0);
signal eqdiffsign, eqdiffsign_d1 :  std_logic;
signal expFrac :  std_logic_vector(10 downto 0);
signal stk :  std_logic;
signal rnd :  std_logic;
signal grd :  std_logic;
signal lsb :  std_logic;
signal addToRoundBit :  std_logic;
signal RoundedExpFrac :  std_logic_vector(10 downto 0);
signal upExc, upExc_d1 :  std_logic_vector(1 downto 0);
signal fracR, fracR_d1 :  std_logic_vector(3 downto 0);
signal expR, expR_d1 :  std_logic_vector(3 downto 0);
signal exExpExc :  std_logic_vector(3 downto 0);
signal excRt2 :  std_logic_vector(1 downto 0);
signal excR :  std_logic_vector(1 downto 0);
signal signR2 :  std_logic;
signal computedR :  std_logic_vector(10 downto 0);
signal X_d1 :  std_logic_vector(4+4+2 downto 0);
signal Y_d1 :  std_logic_vector(4+4+2 downto 0);
begin
   process(clk)
      begin
         if clk'event and clk = '1' then
            eXmeY_d1 <=  eXmeY;
            eYmeX_d1 <=  eYmeX;
            swap_d1 <=  swap;
            newX_d1 <=  newX;
            newX_d2 <=  newX_d1;
            expX_d1 <=  expX;
            expX_d2 <=  expX_d1;
            signX_d1 <=  signX;
            EffSub_d1 <=  EffSub;
            EffSub_d2 <=  EffSub_d1;
            EffSub_d3 <=  EffSub_d2;
            EffSub_d4 <=  EffSub_d3;
            EffSub_d5 <=  EffSub_d4;
            EffSub_d6 <=  EffSub_d5;
            sXsYExnXY_d1 <=  sXsYExnXY;
            excRt_d1 <=  excRt;
            excRt_d2 <=  excRt_d1;
            excRt_d3 <=  excRt_d2;
            excRt_d4 <=  excRt_d3;
            excRt_d5 <=  excRt_d4;
            excRt_d6 <=  excRt_d5;
            signR_d1 <=  signR;
            signR_d2 <=  signR_d1;
            signR_d3 <=  signR_d2;
            signR_d4 <=  signR_d3;
            signR_d5 <=  signR_d4;
            shiftedFracY_d1 <=  shiftedFracY;
            extendedExpInc_d1 <=  extendedExpInc;
            extendedExpInc_d2 <=  extendedExpInc_d1;
            extendedExpInc_d3 <=  extendedExpInc_d2;
            eqdiffsign_d1 <=  eqdiffsign;
            upExc_d1 <=  upExc;
            fracR_d1 <=  fracR;
            expR_d1 <=  expR;
            X_d1 <=  X;
            Y_d1 <=  Y;
         end if;
      end process;
-- Exponent difference and swap  --
   excExpFracX <= X(10 downto 9) & X(7 downto 0);
   excExpFracY <= Y(10 downto 9) & Y(7 downto 0);
   eXmeY <= ("0" & X(7 downto 4)) - ("0" & Y(7 downto 4));
   eYmeX <= ("0" & Y(7 downto 4)) - ("0" & X(7 downto 4));
   swap <= '0' when excExpFracX >= excExpFracY else '1';
   ----------------Synchro barrier, entering cycle 1----------------
   newX <= X_d1 when swap_d1 = '0' else Y_d1;
   newY <= Y_d1 when swap_d1 = '0' else X_d1;
   expX<= newX(7 downto 4);
   excX<= newX(10 downto 9);
   excY<= newY(10 downto 9);
   signX<= newX(8);
   signY<= newY(8);
   EffSub <= signX xor signY;
   sXsYExnXY <= signX & signY & excX & excY;
   sdExnXY <= excX & excY;
   fracY <= "00000" when excY="00" else ('1' & newY(3 downto 0));
   with sXsYExnXY select 
   excRt <= "00" when "000000"|"010000"|"100000"|"110000",
      "01" when "000101"|"010101"|"100101"|"110101"|"000100"|"010100"|"100100"|"110100"|"000001"|"010001"|"100001"|"110001",
      "10" when "111010"|"001010"|"001000"|"011000"|"101000"|"111000"|"000010"|"010010"|"100010"|"110010"|"001001"|"011001"|"101001"|"111001"|"000110"|"010110"|"100110"|"110110", 
      "11" when others;
   ----------------Synchro barrier, entering cycle 2----------------
   signR<= '0' when (sXsYExnXY_d1="100000" or sXsYExnXY_d1="010000") else signX_d1;
   ---------------- cycle 0----------------
   ----------------Synchro barrier, entering cycle 1----------------
   expDiff <= eXmeY_d1 when swap_d1 = '0' else eYmeX_d1;
   shiftedOut <= '1' when (expDiff > 6) else '0';
   shiftVal <= expDiff(2 downto 0) when shiftedOut='0' else CONV_STD_LOGIC_VECTOR(7,3) ;
   RightShifterComponent: RightShifter_5_by_max_7_F800_uid4  -- pipelineDepth=1 maxInDelay=0
      port map ( clk  => clk,
                 rst  => rst,
                 R => shiftedFracY,
                 S => shiftVal,
                 X => fracY);
   ----------------Synchro barrier, entering cycle 2----------------
   ----------------Synchro barrier, entering cycle 3----------------
   sticky <= '0' when (shiftedFracY_d1(4 downto 0)=CONV_STD_LOGIC_VECTOR(0,5)) else '1';
   ---------------- cycle 2----------------
   ----------------Synchro barrier, entering cycle 3----------------
   fracYfar <= "0" & shiftedFracY_d1(11 downto 5);
   EffSubVector <= (7 downto 0 => EffSub_d2);
   fracYfarXorOp <= fracYfar xor EffSubVector;
   fracXfar <= "01" & (newX_d2(3 downto 0)) & "00";
   cInAddFar <= EffSub_d2 and not sticky;
   fracAdder: IntAdder_8_f800_uid8  -- pipelineDepth=0 maxInDelay=0
      port map ( clk  => clk,
                 rst  => rst,
                 Cin => cInAddFar,
                 R => fracAddResult,
                 X => fracXfar,
                 Y => fracYfarXorOp);
   fracGRS<= fracAddResult & sticky; 
   extendedExpInc<= ("00" & expX_d2) + '1';
   LZC_component: LZCShifter_9_to_9_counting_16_F800_uid16  -- pipelineDepth=3 maxInDelay=0
      port map ( clk  => clk,
                 rst  => rst,
                 Count => nZerosNew,
                 I => fracGRS,
                 O => shiftedFrac);
   ----------------Synchro barrier, entering cycle 6----------------
   updatedExp <= extendedExpInc_d3 - ("00" & nZerosNew);
   eqdiffsign <= '1' when nZerosNew="1111" else '0';
   expFrac<= updatedExp & shiftedFrac(7 downto 3);
   ---------------- cycle 6----------------
   stk<= shiftedFrac(1) or shiftedFrac(0);
   rnd<= shiftedFrac(2);
   grd<= shiftedFrac(3);
   lsb<= shiftedFrac(4);
   addToRoundBit<= '0' when (lsb='0' and grd='1' and rnd='0' and stk='0')  else '1';
   roundingAdder: IntAdder_11_f800_uid20  -- pipelineDepth=0 maxInDelay=0
      port map ( clk  => clk,
                 rst  => rst,
                 Cin => addToRoundBit,
                 R => RoundedExpFrac,
                 X => expFrac,
                 Y => "00000000000");
   ---------------- cycle 6----------------
   upExc <= RoundedExpFrac(10 downto 9);
   fracR <= RoundedExpFrac(4 downto 1);
   expR <= RoundedExpFrac(8 downto 5);
   ----------------Synchro barrier, entering cycle 7----------------
   exExpExc <= upExc_d1 & excRt_d6;
   with (exExpExc) select 
   excRt2<= "00" when "0000"|"0100"|"1000"|"1100"|"1001"|"1101",
      "01" when "0001",
      "10" when "0010"|"0110"|"1010"|"1110"|"0101",
      "11" when others;
   excR <= "00" when (eqdiffsign_d1='1' and EffSub_d6='1' and not(excRt_d6="11")) else excRt2;
   signR2 <= '0' when (eqdiffsign_d1='1' and EffSub_d6='1') else signR_d5;
   computedR <= excR & signR2 & expR_d1 & fracR_d1;
   R <= computedR;
end architecture;

