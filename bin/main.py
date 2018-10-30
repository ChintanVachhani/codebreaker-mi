from codebreaker_cv import *


def main():
    image = cv2.imread(sys.argv[1], 1)
    obj = PuzzleDetection()
    success, data = obj.detectSudokuPuzzle(image, int(sys.argv[2]))
    if success:
        print(data)
        # success, filledImage = obj.fillSudokuPuzzle(image, data, int(sys.argv[2]))
        # if success:
        #     cv2.imshow('Puzzle', cv2.resize(filledImage, (600, 600)))
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        # else:
        #     print('Error occurred while filling the image with solution!')
    else:
        print('Invalid image!')
    # obj = CharacterRecognitionWithKNN()
    # obj.generateDataForCharacterRecognitionUsingKNN()


if __name__ == "__main__":
    main()
