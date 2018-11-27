class Hyperparams:
    # data
    train_fpath = 'sudoku.csv'
    test_fpath = 'test.csv'
    
    # model
    num_blocks = 10
    num_filters = 512
    filter_size = 3
    
    # training scheme
    lr = 0.0001
    logdir = "logdir"
    batch_size = 20
    num_epochs = 3
    
