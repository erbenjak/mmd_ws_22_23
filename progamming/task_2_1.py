import numpy as np


def calculate_simularity_cosine():
    # -- a) --
    alpha = 1
    beta = 1

    system1 = np.array([3.06, 500 * alpha, 6 * beta])
    system2 = np.array([2.68, 320 * alpha, 4 * beta])
    system3 = np.array([2.92, 640 * alpha, 6 * beta])

    cosine_sim_1_2 = np.dot(system1, system2) / (np.linalg.norm(system1) * np.linalg.norm(system2))
    cosine_sim_1_3 = np.dot(system1, system3) / (np.linalg.norm(system1) * np.linalg.norm(system3))
    cosine_sim_2_3 = np.dot(system2, system3) / (np.linalg.norm(system2) * np.linalg.norm(system3))

    print('Task 1:')
    print('Cosine similarity between 1 and 2 --- with alpha = ' + str(alpha) + ' and beta = ' + str(beta) + ': ' + str(
        cosine_sim_1_2))
    print('Cosine similarity between 1 and 3 --- with alpha = ' + str(alpha) + ' and beta = ' + str(beta) + ': ' + str(
        cosine_sim_1_3))
    print('Cosine similarity between 2 and 3 --- with alpha = ' + str(alpha) + ' and beta = ' + str(beta) + ': ' + str(
        cosine_sim_2_3))
    print('\n')

    # -- b) --
    alpha = 0.01
    beta = 0.5

    system1_scaled = np.array([3.06, 500 * alpha, 6 * beta])
    system2_scaled = np.array([2.68, 320 * alpha, 4 * beta])
    system3_scaled = np.array([2.92, 640 * alpha, 6 * beta])

    cosine_sim_1_2_scaled = np.dot(system1_scaled, system2_scaled) / (
            np.linalg.norm(system1_scaled) * np.linalg.norm(system2_scaled))
    cosine_sim_1_3_scaled = np.dot(system1_scaled, system3_scaled) / (
            np.linalg.norm(system1_scaled) * np.linalg.norm(system3_scaled))
    cosine_sim_2_3_scaled = np.dot(system2_scaled, system3_scaled) / (
            np.linalg.norm(system2_scaled) * np.linalg.norm(system3_scaled))

    print('Task 2')
    print('Cosine similarity between 1 and 2 --- with alpha = ' + str(alpha) + ' and beta = ' + str(beta) + ': ' + str(
        cosine_sim_1_2_scaled))
    print('Cosine similarity between 1 and 3 --- with alpha = ' + str(alpha) + ' and beta = ' + str(beta) + ': ' + str(
        cosine_sim_1_3_scaled))
    print('Cosine similarity between 2 and 3 --- with alpha = ' + str(alpha) + ' and beta = ' + str(beta) + ': ' + str(
        cosine_sim_2_3_scaled))
    print('\n')

    # -- c) --
    alpha_fair = 1 / ((500 + 320 + 640) / 3)
    beta_fair = 1 / ((6 + 4 + 6) / 3)

    print('Task 3')

    print('The fair alpha value is: ' + str(alpha_fair))
    print('The fair beta value is: ' + str(beta_fair))

    system1_fair = np.array([3.06, 500 * alpha_fair, 6 * beta_fair])
    system2_fair = np.array([2.68, 320 * alpha_fair, 4 * beta_fair])
    system3_fair = np.array([2.92, 640 * alpha_fair, 6 * beta_fair])

    cosine_sim_1_2_fair = np.dot(system1_fair, system2_fair) / (
            np.linalg.norm(system1_fair) * np.linalg.norm(system2_fair))
    cosine_sim_1_3_fair = np.dot(system1_fair, system3_fair) / (
            np.linalg.norm(system1_fair) * np.linalg.norm(system3_fair))
    cosine_sim_2_3_fair = np.dot(system2_fair, system3_fair) / (
            np.linalg.norm(system2_fair) * np.linalg.norm(system3_fair))

    print('Cosine similarity between 1 and 2 --- with alpha = ' + str(alpha_fair) + ' and beta = ' + str(
        beta_fair) + ': ' + str(
        cosine_sim_1_2_fair))
    print('Cosine similarity between 1 and 3 --- with alpha = ' + str(alpha_fair) + ' and beta = ' + str(
        beta_fair) + ': ' + str(
        cosine_sim_1_3_fair))
    print('Cosine similarity between 2 and 3 --- with alpha = ' + str(alpha_fair) + ' and beta = ' + str(
        beta_fair) + ': ' + str(
        cosine_sim_2_3_fair))
    print('\n')


if __name__ == '__main__':
    calculate_simularity_cosine()
