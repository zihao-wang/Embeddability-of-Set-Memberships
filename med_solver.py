"""
In this file, we construct a cmd interface to solve the
minimal embeddable dimension.
"""

import argparse
import os
import matplotlib.pyplot as plt

from routines import experiments

def main():
    parser = argparse.ArgumentParser(
        description='Solve the minimal embeddable dimension')
    parser.add_argument('-M', '--m', type=int, help='Number of elements in a set')
    parser.add_argument('-K', '--k', type=int, help='Number of sets')
    parser.add_argument('-E', '--embedding_name', type=str,
                        help='Name of the embedding module')
    parser.add_argument('-F', '--functional_name', type=str,
                        help='Name of the functional')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs')
    parser.add_argument('--save_dir', type=str, default='logs')
    parser.add_argument('--cuda', type=int, default=0)
    args = parser.parse_args()

    m = args.m
    k = args.k
    embedding_name = args.embedding_name
    functional_name = args.functional_name
    epochs = args.epochs

    os.makedirs(args.save_dir, exist_ok=True)


    # inplace functional

    def plot_log_dict(log_dict):

        fig, ax = plt.subplots(2, 1, figsize=(10, 10))

        ax[0].plot(log_dict['epochs'], log_dict['loss'])
        ax[0].set_title('Loss')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')

        ax[1].plot(log_dict['epochs'], log_dict['sat'])
        ax[1].set_title('Satisfaction')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Satisfaction')

        fig.savefig('logs/emb={}-functional={}-m={}-k={}.png'.format(
            embedding_name, functional_name, m, k
        ))


    # determine the minimal n that satisfied with binary search.
    # we will start with n = m/2

    left = 1
    right = m
    mid = (left + right) // 2

    while True:
        print(f"Trying n = {mid}")
        sat, log_dict = experiments(
            m, mid, k, embedding_name, functional_name, epochs,
            device='cuda:{}'.format(args.cuda) if args.cuda >= 0 else 'cpu'
        )

        plot_log_dict(log_dict)

        if sat:
            print(f"Found n = {mid} that satisfies the condition, decreasing n")

            # decrease n
            right = mid
            mid = (left + right) // 2
        else:
            print(f"n = {mid} does not satisfy the condition, increasing n")

            # increase n
            left = mid
            mid = (left + right) // 2
            if mid == left:
                mid += 1

        if left == right:
            break


    print(f"<Result> MED(m={m}, k={k}) < {mid}")

if __name__ == '__main__':
    main()
