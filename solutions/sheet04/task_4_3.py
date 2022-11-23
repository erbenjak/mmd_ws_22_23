import numpy as np
import numpy.linalg as alg


def google_page_rank_a(dense_input_matrix, beta, eta, print = False):
    primary_size = dense_input_matrix.shape[0]
    # this creates a start vector for the power iteration with 1/N as values
    last_rank_vector = np.reshape((np.ones(primary_size)* (1/primary_size)),(primary_size,-1))
    last_threshhold = eta + 1

    while last_threshhold >= eta:
        new_rank_vetor = dense_input_matrix @ last_rank_vector
        last_threshhold = alg.norm((last_rank_vector - new_rank_vetor), 1)
        if print:
            print("current distance: " + str(last_threshhold))
        last_rank_vector = new_rank_vetor

    # now we need to accommodate for the random teleports
    random_tp_prob_vector = np.reshape((np.ones(primary_size) * (1-beta) * (1/primary_size)),(primary_size,-1))
    rank_vector_google = (last_rank_vector * beta) + random_tp_prob_vector
    return rank_vector_google

def clique_generation(size_N):
    # as each of the N nodes has n-1 putgoing edges those all share the same importance
    primary_importance = 1 / (size_N - 1)
    matrix_unscaled = np.reshape(np.ones(size_N*size_N),(size_N,size_N))
    np.fill_diagonal(matrix_unscaled, 0)
    matrix_scaled = matrix_unscaled * primary_importance
    return matrix_scaled


if __name__ == '__main__':
    print("Starting the page rank routine a) --- for some custom chosen inputs:")

    matrix_dense = np.array([[0.3, 0.5, 0.75], [0.3, 0, 0.125], [0.4, 0.5, 0.125]])
    print("The following input matrix is used: \n" + str(matrix_dense))

    beta = 0.99
    print("The following input beta (no teleport probability) is used:\n " + str(beta))

    eta = 0.01
    print("The following eta is used: " + str(eta))

    page_rank_vector = google_page_rank_a(matrix_dense, beta, eta)
    print("The following pagerank-vector was calculated:\n" + str(page_rank_vector))

    #statring with task b)
    print("\nCreating some clique dense matricies as defined in b):\n")
    clique_of_size_4 = clique_generation(4)
    print("Generated clique of size 4:\n")
    print(clique_of_size_4)

    clique_of_size_6 = clique_generation(6)
    print("\nGenerated clique of size 6:\n")
    print(clique_of_size_6)

    beta = 0.8
    print("\nThe following input beta (no teleport probability) is used: " + str(beta))

    eta = (1/12)
    print("The following eta is used: " + str(eta))

    page_rank_vector_clique_4 = google_page_rank_a(clique_of_size_4, beta, eta)
    print("The following pagerank-vector was calculated (clique size 4):\n" + str(page_rank_vector_clique_4))

    page_rank_vector_clique_6 = google_page_rank_a(clique_of_size_6, beta, eta)
    print("\nThe following pagerank-vector was calculated (clique size 6):\n" + str(page_rank_vector_clique_6))

