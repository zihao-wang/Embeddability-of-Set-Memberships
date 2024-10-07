import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import trange
from dataset import SetMembershipDataset
from embedding import get_embedding_module
from functionals import get_functional


def init_tol(dataset):
    tol = 10000 // len(dataset) + 10
    return max(tol, 10)


def experiments(m, n, k, embedding_name, functional_name, epochs=1000):
    dataset = SetMembershipDataset(m, k)
    loader = DataLoader(dataset, batch_size=1, num_workers=8, shuffle=True)

    Embedding = get_embedding_module(embedding_name)
    model = Embedding(m, n, len(dataset))

    if torch.cuda.is_available():
        model.cuda()

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    epoch_list = []
    sat_list = []
    loss_list = []

    track_log = []

    tol = init_tol(dataset)

    functional = get_functional(functional_name)

    with trange(epochs, desc=f"m={m}, n={n}, k={k}") as t:
        for e in t:
            satisfied = 0
            loss_sum = 0
            for ele_mask, set_id in loader:
                if torch.cuda.is_available():
                    ele_mask, set_id = ele_mask.cuda(), set_id.cuda()

                opt.zero_grad()
                pele_emb, nele_emb, set_emb = model(ele_mask, set_id)

                assert pele_emb.size(0) == ele_mask.sum()
                assert nele_emb.size(0) + pele_emb.size(0) == ele_mask.size(0)

                pos_metric = functional(pele_emb, set_emb)
                neg_metric = functional(nele_emb, set_emb)

                pos_loss = pos_metric.view(-1, 1)
                neg_loss = neg_metric.view(1, -1)

                loss = torch.relu(pos_loss - neg_loss).mean()

                satisfied += float(pos_metric.max() < neg_metric.min())
                loss_sum += loss.item()

                loss.backward()
                opt.step()

            sat = satisfied / len(dataset)
            loss = loss_sum / len(dataset)

            sat_list.append(sat)
            epoch_list.append(e)
            loss_list.append(loss)

            if sat_list and sat <= max(sat_list):
                tol -= 1
                if tol < 0:
                    break
            elif sat == 1:
                break
            else:
                tol = init_tol()

            t.set_postfix({"sat ratio": sat, "loss": loss})

        finally_sat = max(sat_list) > 1-1e-6
    return finally_sat, {"epochs": epoch_list, "sat": sat_list, "loss": loss_list}
