class params:
    # data
    trainFilePath = 'sudoku.csv'
    testFilePath = 'test.csv'

    # model
    num_blocks = 10
    num_filters = 512
    filter_size = 3

    # training scheme
    lr = 0.0001
    modelDir = "model"
    batch_size = 20
    num_epochs = 3
