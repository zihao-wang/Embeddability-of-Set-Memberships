import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import trange
from dataset import SetMembershipDataset
from embedding import get_embedding_module
from functionals import get_functional


def init_tol(dataset):
    tol = 10000 // len(dataset) + 10
    return max(tol, 20)


def experiments(m, n, k, embedding_name, functional_name, epochs=2000, batch_size=512, device='cpu'):
    dataset = SetMembershipDataset(m, k)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=True)

    Embedding = get_embedding_module(embedding_name)
    model = Embedding(m, n, len(dataset))

    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    epoch_list = []
    sat_list = []
    loss_list = []

    tol = init_tol(loader)

    functional = get_functional(functional_name)

    with trange(epochs, desc=f"m={m}, n={n}, k={k}") as t:
        for e in t:
            satisfied = 0
            loss_sum = 0
            for ele_mask, set_id in loader:
                ele_mask, set_id = ele_mask.to(device), set_id.to(device)

                # ele_mask [batch, m]
                # set_id [batch]
                opt.zero_grad()

                set_emb = model(set_id) # [batch, n]
                ele_emb = model.ele_emb.weight # [m, n]

                # ele_mask is a boolean mask of batch sizes
                # apply it to ele_emb per batch, and gather it into pele_emb
                # the shape of pele_emb is [batches of positive samples, n]
                indices = ele_mask.nonzero(as_tuple=False)
                pele_emb = ele_emb[indices[:, 1]]  # use only the relevant indices for selection
                num_pele_per_batch = ele_mask.sum(1)
                set_emb_p = set_emb.repeat_interleave(num_pele_per_batch, dim=0)

                # apply the negation of ele_mask to ele_emb per batch,
                # and gather it into nele_emb
                # the shape of nele_emb is [batches of negative samples, n]
                indices = (~ele_mask).nonzero(as_tuple=False)
                nele_emb = ele_emb[indices[:, 1]]  # use only the relevant indices for selection
                num_nele_per_batch = (~ele_mask).sum(1)
                set_emb_n = set_emb.repeat_interleave(num_nele_per_batch, dim=0)

                pos_metric = functional(pele_emb, set_emb_p)
                max_pos_metric_per_batch = torch.cat([
                    m.max().reshape(1,)
                    for m in torch.split(pos_metric, num_pele_per_batch.tolist())
                ])
                neg_metric = functional(nele_emb, set_emb_n)
                min_neg_metric_per_batch = torch.cat([
                    m.min().reshape(1,)
                    for m in torch.split(neg_metric, num_nele_per_batch.tolist())
                ])

                loss = torch.relu(max_pos_metric_per_batch - min_neg_metric_per_batch).mean()

                satisfied += torch.mean((max_pos_metric_per_batch < min_neg_metric_per_batch).float()).item()
                loss_sum += loss.item()

                loss.backward()
                opt.step()

            sat = satisfied / len(loader)
            loss = loss_sum / len(loader)

            prev_max = max(sat_list) if sat_list else 0

            sat_list.append(sat)
            epoch_list.append(e)
            loss_list.append(loss)

            if sat_list and sat <= prev_max:
                tol -= 1
                if tol < 0:
                    break
            elif sat > 1 - 1e-6:
                break
            else:
                tol = init_tol(loader)

            t.set_postfix({"sat ratio": sat, "loss": loss})

        finally_sat = max(sat_list) > 1-1e-6
    return finally_sat, {"epochs": epoch_list, "sat": sat_list, "loss": loss_list}
