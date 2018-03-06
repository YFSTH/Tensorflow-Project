from selectProposals import selectProposals
import pickle

with open('proposal_debugging.pkl', 'rb') as file:
    logits_, train_proposals_img, train_ground_truth_tensor, train_selection_tensor = pickle.load(file)

proposal_selection_tensor = selectProposals(iou_threshold=0, max_n_highest_cls_scores=200, logits=logits_,
                                            proposal_tensor=train_proposals_img,
                                            ground_truth_tensor=train_ground_truth_tensor,
                                            selection_tensor=train_selection_tensor, training=True)