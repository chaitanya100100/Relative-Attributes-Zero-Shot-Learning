"""
----------------
8 categories
----------------
1. AlexRodriguez
2. CliveOwen
3. HughLaurie
4. JaredLeto
5. MileyCyrus
6. ScarlettJohansson
7. ViggoMortensen
8. ZacEfron

----------------
11 attributes
----------------
1. 'Male',
2. 'White',
3. 'Young',
4. 'Smiling',
5. 'Chubby',
6. 'VisibleForehead',
7. 'BushyEyebrows',
8. 'NarrowEyes',
9. 'PointyNose',
10. 'BigLips',
11. 'RoundFace'

     1   2   3   4   5   6   7   8

 1   6   8   7   5   2   1   4   3
 2   1   2   3   5   7   6   8   4
 3   5   3   2   4   8   6   1   7
 4   4   4   3   1   6   5   2   5
 5   8   4   3   2   6   7   1   5
 6   5   5   5   1   3   4   5   2
 7   6   7   5   8   1   2   4   3
 8   4   6   5   2   1   3   7   8
 9   1   2   8   3   3   4   3   7
10   7   5   1   2   6   8   3   4
11   6   4   1   3   8   7   2   5

"""
seen = [1, 2, 3, 4]
unseen = [7, 8]

relative_input = {
    # 5
    5 : [
        [-1, 4], #1
        [4, -1], #2
        [-1, 3], #3
        [4, 3], #4
        [-1, 4], #5
        [4, -1], #6
        [-1, 3], #7
        [2, -1], #8
        [2, 3], #9
        [4, 2], #10
        [3, 4] #11
    ],
    # 7
    7 : [
        [-1, 4], #1
        [4, -1], #2
        [-1, 3], #3
        [4, 3], #4
        [-1, 4], #5
        [4, -1], #6
        [-1, 3], #7
        [2, -1], #8
        [2, 3], #9
        [4, 2], #10
        [3, 4] #11
    ],
    # 8
    8 : [
        [-1, 4], #1
        [3, 4], #2
        [1, -1], #3
        [1, -1], #4
        [2, 1], #5
        [4, 3], #6
        [-1, 3], #7
        [2, -1], #8
        [4, 3], #9
        [4, 2], #10
        [2, 1] #11
    ]
}

zero_shot_weights_directory = "../saved_data/zero_shot_weights_directory"
