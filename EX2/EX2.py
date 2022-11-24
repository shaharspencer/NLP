H_GIVEN_H = 0.5
L_GIVEN_H = 0.5
L_GIVEN_L =0.6
H_GIVEN_L =0.4




H_DICT = {"A":0.2, "C": 0.3, "G": 0.3, "T":0.2}
L_DICT = {"A":0.3, "C": 0.2, "G": 0.2, "T":0.3}
def VitarbiGenes(s):
    DP = [[(0,None) for i in range(2)] for j in range(len(s))]
    for i in range(len(s)):
        if i == 0:
            for j in range(2):
                if j == 0:
                    DP[i][j] =  ("H", H_DICT[s[i]] * H_GIVEN_H)

                elif j == 1:
                    DP[i][j] = ("H", L_DICT[s[i]] * L_GIVEN_H)
        else:
            for j in range(2):
                if j == 0:
                    if DP[i-1][0][1]*H_DICT[s[i]]*H_GIVEN_H >\
                            DP[i-1][1][1]*H_DICT[s[i]]*H_GIVEN_L:
                        DP[i][j] = ("H", DP[i-1][0][1]*H_DICT[s[i]]*H_GIVEN_H)
                    else:
                        DP[i][j] = ("L", DP[i-1][1][1]*H_DICT[s[i]]*H_GIVEN_L)


                elif j == 1:
                    if DP[i - 1][0][1] * L_DICT[s[i]] * L_GIVEN_H > DP[i - 1][1][1] * L_DICT[s[i]] * L_GIVEN_L:
                        DP[i][j] = ("H", DP[i - 1][0][1] * L_DICT[s[i]] * L_GIVEN_H)
                    else:
                        DP[i][j] = ("L", DP[i - 1][1][1] * L_DICT[s[i]] * L_GIVEN_L)


    for r in DP:
        print(r)


s= "ACCGTCCA"
VitarbiGenes(s)



print(max([1.0934999999999998e-07, 1.6402499999999996e-07]))