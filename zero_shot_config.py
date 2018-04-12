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
"""
seen = [1, 2, 3, 4]
unseen = [7, 8]

relative_input = {
    # 7
    7 : [
        [-1, 1], #1
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

zero_shot_weights_directory = "zero_shot_weights_directory"
