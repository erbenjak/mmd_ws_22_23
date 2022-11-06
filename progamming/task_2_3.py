import numpy as np


def calculate_simularity_cosine():
    user_1 = np.array([0.77, 1.77, 0, 1.77, -2.33, 0, -0.33, -1.33])
    user_2 = np.array([0, 0.77, 1.77, 0.77, -1.33, -0.33, -1.33, 0])
    user_3 = np.array([-1, 0, -2, 0, 0, 1, 2, 0 ])

    cosine_sim_1_2 = np.dot(user_1, user_2) / (np.linalg.norm(user_1) * np.linalg.norm(user_2))
    cosine_sim_1_3 = np.dot(user_1, user_3) / (np.linalg.norm(user_1) * np.linalg.norm(user_3))
    cosine_sim_2_3 = np.dot(user_2, user_3) / (np.linalg.norm(user_2) * np.linalg.norm(user_3))

    print('Task 1:')
    print('Cosine similarity between 1 and 2 : ' + str(
        cosine_sim_1_2))
    print('Cosine similarity between 1 and 3 : ' + str(
        cosine_sim_1_3))
    print('Cosine similarity between 2 and 3 : ' + str(
        cosine_sim_2_3))
    print('\n')


if __name__ == '__main__':
    calculate_simularity_cosine()
