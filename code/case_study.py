import pickle
import utils

if __name__ == '__main__':
    mat = pickle.load(open('../FewShotWithoutForgetting/datasets/MiniImagenet/miniImageNet_category_split_train_phase_train.pickle', 'rb'))
    # utils.show_img(mat)
    """
    case_ids = range(10)
    cases = []
    for i in case_ids:
        case = pickle.load(open('../case_study/case' + str(i) + '.pkl', 'rb'))
        cases.append(case)

    case_id = 9
    print(cases[case_id]['pred_labels'], cases[case_id]['gt_labels'])
    utils.show_six_images(cases[case_id]['data'])
    """
