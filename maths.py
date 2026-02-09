import numpy as np
import torch
from torch.nn import functional as F


def distance_l2(ref_embeds: torch.Tensor, other_embeds: torch.Tensor) -> torch.Tensor:
    """
    Computes the L2 (euclidean) distance between two sets of embeddings.

    To prevent CUDA out-of-memory issues, the computation cannot be made at once. Here, one computation
    will be performed for each OTHER embeddings to be tested: this assumes that you don't provide
    too many REFERENCE embeddings (thousands is definitely OK for any GPU, not sure for millions), but the
    number of OTHER embeddings can be arbitrarily large.

    :return: The matrix of distances between the two embeddings - references as lines, others as columns.
    """
    assert ref_embeds.dim() == other_embeds.dim() == 2 and ref_embeds.shape[1] == other_embeds.shape[1]
    distances = torch.zeros((ref_embeds.shape[0], other_embeds.shape[0]), device=ref_embeds.device)
    # First, compute sums of squares for each line (that's what can't be done at once for memory issues)
    for i in range(other_embeds.shape[0]):
        distances[:, i] = torch.sum(torch.square(ref_embeds - other_embeds[i:i+1, :]), dim=1)  # Fill col by col
    # Finally compute the sqrt only once
    return torch.sqrt(distances)


def distance_cosine(ref_embeds: torch.Tensor, other_embeds: torch.Tensor) -> torch.Tensor:
    """
    Computes the cosine distance (= 1 - similarity) between two sets of embeddings.

    :return: The matrix of distances between the two embeddings - references as lines, others as columns.
    """
    assert ref_embeds.dim() == other_embeds.dim() == 2 and ref_embeds.shape[1] == other_embeds.shape[1]
    # Normalize embeddings before computing dot products
    ref_embeds_normalized = ref_embeds / ref_embeds.norm(dim=1, keepdim=True)
    other_embeds_normalized = other_embeds / other_embeds.norm(dim=1, keepdim=True)
    # Compute the cosine similarity using matrix multiplication - this could lead to CUDA out-of-memory....?
    cosine_similarity = torch.mm(ref_embeds_normalized, other_embeds_normalized.transpose(0, 1))
    return 1.0 - cosine_similarity


def distance_mahalanobis(ref_embeds: torch.Tensor, other_embeds: torch.Tensor) -> torch.Tensor:
    """
    Computes the Mahalanobis distance (= 1 - similarity) between two sets of embeddings.

    The covariance matrix is currently estimated using reference embeddings only - this is not representative of
     the whole dataset (but it would be tedious to retrieve proper stats for all training embeddings computed
     by a given model....

    :param ref_embeds:
    :param other_embeds:
    :return: The matrix of distances between the two embeddings - references as lines, others as columns.
    """
    # Compute the covariance matrix of the reference embeddings - and its inverse now
    ref_embeds_centered = ref_embeds - torch.mean(ref_embeds, dim=0)
    covariance_matrix = torch.mm(ref_embeds_centered.T, ref_embeds_centered) / (ref_embeds.size(0) - 1)
    inv_covariance_matrix = torch.inverse(covariance_matrix)

    # Compute the Mahalanobis distance - row by row (otherwise: probable CUDA out of memory, same as L2)
    distance_matrix = torch.zeros((ref_embeds.shape[0], other_embeds.shape[0]), device=ref_embeds.device)
    for i in range(other_embeds.shape[0]):  # for each "other" embedding, sequentially
        other_embed = other_embeds[i, :]
        # Here, diff has 1 vector on each line ; so that corresponds to an "already transposed" (x-y).T in the
        #  original distance formulation
        # To get a matrix of (x-y) (not transposed) where each (x-y) is a column vector, use diff.T (not needed here)
        diff = ref_embeds - other_embed  # w/ broadcasting. Output shape: N x h (N = nb of ref items, h = embed dims)
        scaled_diffs = torch.mm(diff, inv_covariance_matrix)  # N x h shape also, scaled diffs on each line
        dot_products = torch.sum(scaled_diffs * diff, dim=1)  # Shape: N x h before summation, N after summation
        distance_matrix[:, i] = torch.sqrt(dot_products)

    return distance_matrix


def triplet_loss_from_pairs(embeddings_0: torch.tensor, embeddings_1: torch.tensor, margin=1.0, p=2):
    """
    Computes the triplet loss for two paired sets of embeddings (embeddings_0, embeddings_1) where rows at the
    same index must be positives, while rows at different indices must be negatives.

    :param embeddings_0: shape N x D
    :param embeddings_1: shape N x D
    :param margin: see https://docs.pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html
    :param p: see https://docs.pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html
    :return:
    """
    assert embeddings_0.shape == embeddings_1.shape and len(embeddings_0.shape) == 2
    N = embeddings_0.shape[0]  # Number of pairs... corresponds to half the actual minibatch size
    N_negatives_by_pair = 2 * (N-1)  # Number of available negatives for each pair of positives
    D = embeddings_0.shape[1]  # Embedding vector length

    #  Solution w/ one triplet loss call:
    #     needs to build large matrices of anchor, positive and negative embeddings
    #  expected size: 3 matrices of size (N.(2N-2)) x D,
    #     e.g. for hidden dim 768:
    #                batch size 32 :  480 x 768 matrices
    #                batch size 128: 8064 x 768 matrices
    # Build the positives, pair-by-pair (the 2N-2 first rows are exactly the same rows, etc...)
    anchors = embeddings_0.unsqueeze(1).repeat(1, N_negatives_by_pair, 1).reshape(-1, D)
    positives = embeddings_1.unsqueeze(1).repeat(1, N_negatives_by_pair, 1).reshape(-1, D)
    # Build the negatives, pair-by-pair
    negatives = []
    for i in range(N):
        # Roll the embeddings to move the positives always at first pos (index 0)
        negatives.append(torch.roll(embeddings_0, shifts=-i, dims=0)[1:, :])
        negatives.append(torch.roll(embeddings_1, shifts=-i, dims=0)[1:, :])
    negatives = torch.cat(negatives, dim=0)

    return F.triplet_margin_loss(anchors, positives, negatives, margin=margin, p=p)


def infonce_loss(embeddings_0: torch.Tensor, embeddings_1: torch.Tensor, temperature=0.1):
    """ InfoNCE loss for two sets of paired embeddings (positives must be stored in the same row number
     in the two embeddings matrices _0 and _1) """
    # Normalise embeddings before computing similarity matrix
    embeddings_0 = F.normalize(embeddings_0, p=2, dim=1)
    embeddings_1 = F.normalize(embeddings_1, p=2, dim=1)
    similarity_matrix = torch.matmul(embeddings_0, embeddings_1.T) / temperature

    # Create "labels" to indicate which 'logits' in the matrix should be the biggest
    labels = torch.arange(similarity_matrix.size(0), device=embeddings_0.device)
    # InfoNCE loss, average of the two losses
    loss_0 = F.cross_entropy(similarity_matrix, labels)
    loss_1 = F.cross_entropy(similarity_matrix.T, labels)
    return (loss_0 + loss_1) / 2


def _check_mix_indices(tracks_indices: torch.Tensor):
    """
    Check that tracks indices are always as follows:
       1, 1, ..., 1, -1, 2, 2, ..., 2, -2, 3, 3, ..., 3, -3, ...
    (doing that will ensure easier loss computations). Corresponds to that order:
    instr0,0 instr0,1 instr0,2 mix0 instr1,0 instr1,1 instr1,2 mix1 instr2,0 instr2,1 instr2,2 mix2 ...
    where 'instrX,Y' designate single-instrument tracks (stems) and 'mixX' designate the corresponding mix of stems.
    """
    N = tracks_indices.shape[0]

    assert torch.all(tracks_indices != 0)
    mix_mask = tracks_indices < 0
    mix_indices = tracks_indices[mix_mask].cpu().numpy() * -1
    assert np.all(mix_indices > 0)
    assert np.all(mix_indices == np.arange(1, len(mix_indices) + 1))
    N_mixes = len(mix_indices)

    assert N % N_mixes == 0
    N_stems_per_mix = (N // N_mixes) - 1
    N_stems = N_mixes * N_stems_per_mix

    expected_tracks_indices = sum([[mix_i] * N_stems_per_mix + [-mix_i] for mix_i in range(1, N_mixes+1)], [])
    expected_tracks_indices = torch.tensor(expected_tracks_indices, device=tracks_indices.device)
    assert torch.all(tracks_indices == expected_tracks_indices)

    return N, mix_mask, N_mixes, N_stems_per_mix, N_stems


def infonce_mix_loss(embeddings: torch.Tensor, tracks_indices: torch.Tensor, temperature=0.1):
    """
    The is the "basic" similarity loss for mixes, which considers that each mono track has a
    single positive match: the mix it corresponds to. All other mixes are the negative embeddings.

                 mix0    mix1    mix2...
    instr0,0      +       -        -
    instr0,1      +       -        -
    instr0,2      +       -        -
    instr1,0      -       +        -
    instr1,1      -       +        -
    instr1,2      -       +        -
    instr2,0      -       -        +
    instr2,1      -       -        +
    instr2,2      -       -        +

    As opposed to the "full" loss in infonce_mix_full_loss(...),
    the relationship between stems is just ignored.
    This loss does not enforce the different stems of a mix to be negatives (the relationship between
    those embeddings is just ignored), or stems of different mixes to be negatives.

    :param embeddings: shape N x H, where N is the total number of tracks (stems and mixes) in the minibatch,
        and H is the embedding vector length.
    :param tracks_indices: The same positive index (e.g. +2) for multiple tracks indicate mono-instrument
        tracks (stems) summed together in the same mix. A negative index (e.g. -2) indicates a mix of those stems.
        Null indices (0) must NOT be used.
    """
    assert embeddings.shape[0] == tracks_indices.shape[0]
    N, mix_mask, N_mixes, N_stems_per_mix, N_stems = _check_mix_indices(tracks_indices)

    embeddings = F.normalize(embeddings, p=2, dim=1)
    mixes_embeddings = embeddings[mix_mask, :]  # shape N_mixes x H
    mono_embeddings = embeddings[~mix_mask, :]  # shape N_stems x H

    # Compute similarity between mono tracks and mixed tracks. Shape: N_stems x N_mixes
    similarity_matrix = torch.matmul(mono_embeddings, mixes_embeddings.T) / temperature
    # Labels: each mix (column in the similarity matrix) has multiple corresponding "positive" mono tracks
    labels = tracks_indices[~mix_mask] - 1
    loss = F.cross_entropy(similarity_matrix, labels.to(similarity_matrix.device))
    return loss


def infonce_mix_full_loss(embeddings: torch.Tensor, tracks_indices: torch.Tensor, temperature=0.1, symmetric=True):
    """
    Example: Considering a mix of 3 single-instrument tracks (stems), embeddings have to be stored like this:
    instr0,0 instr0,1 instr0,2 mix0 instr1,0 instr1,1 instr1,2 mix1 instr2,0 instr2,1 instr2,2 mix2 ...
    (this is enforced by the check)

    For the self-similarity matrix, the contrastive positives/negatives will be:
     o - - +   - - - -   - - - -
     - o - +   - - - -   - - - -
     - - o +   - - - -   - - - -
     + + + o   - - - -   - - - -

     - - - -   o - - +   - - - -
     - - - -   - o - +   - - - -
     - - - -   - - o +   - - - -
     - - - -   + + + o   - - - -

     - - - -   - - - -   o - - +
     - - - -   - - - -   - o - +
     - - - -   - - - -   - - o +
     - - - -   - - - -   + + + o
    where the 'o' indicates similarity values that should be ignored.
    We'll just mask the output similarities from the diagonal (self-similarity must not be optimized)

     Can also use a non-symmetric similarity matrix, with mixes' embeddings removes from rows (the CE loss becomes a
     single-class classification problem) - this should not change much, but we can try it
     o - - +    - - - -    - - - -
     - o - +    - - - -    - - - -
     - - o +    - - - -    - - - -
     (Mix0 embedding (and associated target) removed)
     - - - -    o - - +    - - - -
     - - - -    - o - +    - - - -
     - - - -    - - o +    - - - -
     (Mix1 embeddings removed)
     - - - -    - - - -    o - - +
     - - - -    - - - -    - o - +
     - - - -    - - - -    - - o +
     (Mix2 embeddings removed)
     ...
    """
    assert embeddings.shape[0] == tracks_indices.shape[0]
    N, mix_mask, N_mixes, N_stems_per_mix, N_stems = _check_mix_indices(tracks_indices)
    N_tracks_per_mix = N_stems_per_mix + 1

    embeddings = F.normalize(embeddings, p=2, dim=1)
    similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature
    # Force self-similarity to be ignored by settings diagonal coeffs to 0
    similarity_matrix[torch.eye(N).to(torch.bool)] = 0.0

    # Build the matrix of targets on the CPU (many small calls)
    if symmetric:  # Some rows (the Mix ones) have multiple positives (use float targets = class "probabilities")
        targets = torch.eye(N)  # Don't use .to(torch.int64): will be considered as class probabilities
        for mix_i in range(N_mixes):
            single_instrument_tracks_mask = torch.zeros((N, )).to(torch.bool)
            mix_row = ((mix_i + 1) * N_tracks_per_mix) - 1
            single_instrument_tracks_mask[mix_i*N_tracks_per_mix:(mix_i*N_tracks_per_mix + N_stems_per_mix)] = True
            targets[single_instrument_tracks_mask, mix_row] = 1
            targets[mix_row, single_instrument_tracks_mask] = 1
    # Non-symmetric version: can use integer classes
    else:
        similarity_matrix = similarity_matrix[~mix_mask, :]
        targets = torch.ones((N_stems, ),).to(torch.long) * -1
        for mix_i in range(N_mixes):
            mix_col_idx = ((mix_i+1) * N_tracks_per_mix) - 1
            targets[mix_i*N_stems_per_mix : (mix_i+1)*N_stems_per_mix] = mix_col_idx
    targets = targets.to(embeddings.device)

    # As opposed to CLIP/CLAP, similarity matrix is here symmetric (there's only one set of embeddings
    #  use for the rows and the cols) so we don't need to compute 2 oriented CE losses and average them
    loss = F.cross_entropy(similarity_matrix, targets, reduction='none')
    return loss.mean()  # Compute the average here (after a final size sanity check)


def triplet_loss_from_mix(embeddings: torch.tensor,tracks_indices: torch.Tensor, margin=1.0, p=2, full=False):
    """
    See _check_mix_indices(...) documentation for more info about the required tracks_indices and embeddings' structure.

    TODO finish doc

    :param full: If False, only the mixes are used as anchors. If True, all tracks (mixes and stems) are used as
                    anchors at some point.
    """
    assert embeddings.shape[0] == tracks_indices.shape[0]
    N, mix_mask, N_mixes, N_stems_per_mix, N_stems = _check_mix_indices(tracks_indices)
    N_total_tracks_per_mix = (N_stems_per_mix + 1)
    assert N_total_tracks_per_mix == N // N_mixes
    mix_embeds = embeddings[mix_mask, :]
    stems_embeds = embeddings[~mix_mask, :]

    # Base loss: use the mixes as anchors, use a mix' stems as positives and all the rest as negatives
    #      retrieve negatives mix-per-mix... This represents not so many calls (not so many mixes)

    # For each mix (N_mixes total), there's N_stems_per_mix positive stems.
    # For each mix and positive stem, there's:
    # - (N_stems_per_mix - 1) from the current mix
    # - ((N_mixes - 1) * N_total_tracks_per_mix) negative mixes and stems from other mixes
    N_negatives_per_mix_and_stem = (N_stems_per_mix - 1) + ((N_mixes - 1) * N_total_tracks_per_mix)
    # Total number of rows in the matrices fed to triplet_margin_loss:
    #                  anchors *    positives    *              negatives
    N_triplet_embeds = N_mixes * N_stems_per_mix * N_negatives_per_mix_and_stem
    # Example values:
    #      396 triplet embeds for N_mixes = 6  and N_stems_per_mix = 3   (batch size 24)
    #     1656 triplet embeds for N_mixes = 12 and N_stems_per_mix = 3   (batch size 48)

    # Use repeats for the anchors and positives
    anchors = torch.repeat_interleave(mix_embeds, N_triplet_embeds // N_mixes, dim=0)
    positives = torch.repeat_interleave(stems_embeds, N_negatives_per_mix_and_stem, dim=0)

    # for negatives: iterating seems inevitable
    negatives = torch.zeros_like(anchors)
    negative_i = 0  # Next row index to store negatives
    for mix_i in range(N_mixes):
        current_mix_and_stems_mask = torch.zeros((N,)).to(torch.bool).to(embeddings.device)
        current_mix_and_stems_mask[mix_i*N_total_tracks_per_mix:(mix_i+1)*N_total_tracks_per_mix] = True
        current_mix_and_stems_embeds = embeddings[current_mix_and_stems_mask, :]
        current_stems_embeds = current_mix_and_stems_embeds[:-1, :]

        other_mixes_and_stems_embeds = embeddings[~current_mix_and_stems_mask, :]
        for positive_stem_i in range(N_stems_per_mix):
            other_stems_from_current_mix_embeds = current_stems_embeds[torch.arange(N_stems_per_mix) != positive_stem_i, :]

            negatives[negative_i:negative_i+(N_stems_per_mix-1), :] = other_stems_from_current_mix_embeds
            negative_i += (N_stems_per_mix - 1)
            negatives[negative_i:negative_i+other_mixes_and_stems_embeds.shape[0], :] = other_mixes_and_stems_embeds
            negative_i += other_mixes_and_stems_embeds.shape[0]

    mixes_as_anchors_loss = F.triplet_margin_loss(anchors, positives, negatives, margin=margin, p=p, reduction='none')

    if not full:
        return mixes_as_anchors_loss.mean()
    else:
        # Full triplet loss, where all tracks (stem or mix) are considered as anchor at some point.
        #    We just swap the anchors and positives (negatives remain unchanged)
        anchors, positives = positives, anchors  # Stems become anchors
        stems_as_anchors_loss = F.triplet_margin_loss(anchors, positives, negatives, margin=margin, p=p, reduction='none')
        return torch.cat((mixes_as_anchors_loss, stems_as_anchors_loss)).mean()

