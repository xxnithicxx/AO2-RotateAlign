import torch
from ...utils.misc import inverse_sigmoid
from mmrotate.core import obb2poly, poly2obb

def prepare_for_cdn(dn_args, training, num_queries, num_classes, hidden_dim, label_enc):
    """
    A major difference of DINO from DN-DETR is that the author process pattern embedding in its detector
    forward function and use learnable tgt embedding, so we change this function a little bit.
    :param dn_args: targets, dn_number, label_noise_ratio, box_noise_scale
    :param training: if it is training or inference
    :param num_queries: number of queires
    :param num_classes: number of classes
    :param hidden_dim: transformer hidden dim
    :param label_enc: encode labels in dn
    :return:
    """
    if not training:
        return None, None, None, None
    
    targets, dn_number, label_noise_ratio, box_noise_scale, angle_noise_range = dn_args
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Tính toán số lượng positive samples
    known = [torch.ones_like(t["labels"]).to(device) for t in targets]
    batch_size = len(known)
    known_num = [sum(k) for k in known]
    max_known = int(max(known_num))
    
    # Điều chỉnh dn_number
    dn_number = dn_number * 2
    if max_known == 0:
        dn_number = 1
    elif dn_number >= 100:
        dn_number = max(1, dn_number // (max_known * 2))
    elif dn_number < 1:
        dn_number = 1
    
    # Gom nhóm dữ liệu
    unmask_bbox = unmask_label = torch.cat(known)
    labels = torch.cat([t["labels"] for t in targets])
    boxes = torch.cat([t["boxes"] for t in targets])
    img_size = 1024  # Kích thước ảnh
    boxes[:, :4] = boxes[:, :4] / img_size  # Chuyển về [0, 1]
    batch_idx = torch.cat([torch.full_like(t["labels"].long(), i) for i, t in enumerate(targets)])

    # Tạo indices
    known_indice = torch.nonzero(unmask_label + unmask_bbox).view(-1)
    known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
    known_labels = labels.repeat(2 * dn_number, 1).view(-1)
    known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1).to(device)
    known_bboxs = boxes.repeat(2 * dn_number, 1)
    known_labels_expaned = known_labels.clone()
    known_bbox_expand = known_bboxs.clone()
    
    # Thêm nhiễu vào labels nếu cần
    if label_noise_ratio > 0:
        p = torch.rand_like(known_labels_expaned.float())
        chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(-1)  # half of bbox prob
        new_label = torch.randint_like(chosen_indice, 0, num_classes)  # randomly put a new one here
        known_labels_expaned.scatter_(0, chosen_indice, new_label)
    
    # Chuẩn bị padding 
    single_pad = max_known
    pad_size = int(single_pad * 2 * dn_number)
    
    # Tạo positive và negative indices
    positive_idx = torch.arange(len(boxes), device=device).unsqueeze(0).repeat(dn_number, 1)
    positive_idx += (torch.arange(dn_number, device=device) * len(boxes) * 2).unsqueeze(1)
    positive_idx = positive_idx.flatten()
    negative_idx = positive_idx + len(boxes)
    
    
    # Thêm nhiễu vào bounding boxes
    if box_noise_scale > 0:
        # Chuyển đổi từ định dạng (cx, cy, w, h, angle) sang 8 điểm (xyxyxyxy)
        known_bbox_poly = obb2poly(known_bboxs)
        
        # Tính toán tâm của đối tượng
        center_x = (known_bbox_poly[:, 0] + known_bbox_poly[:, 2] + known_bbox_poly[:, 4] + known_bbox_poly[:, 6]) / 4
        center_y = (known_bbox_poly[:, 1] + known_bbox_poly[:, 3] + known_bbox_poly[:, 5] + known_bbox_poly[:, 7]) / 4
        centers = torch.stack([center_x, center_y], dim=1)
        
        # Tính vector từ tâm đến mỗi điểm góc
        vectors = torch.zeros_like(known_bbox_poly)
        for i in range(0, 8, 2):
            vectors[:, i] = known_bbox_poly[:, i] - centers[:, 0]
            vectors[:, i+1] = known_bbox_poly[:, i+1] - centers[:, 1]
        
        # Tạo nhiễu cho các vector (giữ nguyên hình dạng)
        # Tạo 4 cặp nhiễu cho 4 góc
        noise_scale = (torch.ones_like(vectors) * box_noise_scale)
        negative_idx = negative_idx.to('cpu')
        noise_scale[negative_idx] *= 2.0  # Nhiễu lớn hơn cho negative samples
        
        # Tạo nhiễu ngẫu nhiên với dấu
        rand_sign = torch.randint_like(vectors, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
        rand_part = torch.rand_like(vectors)
        
        # Đảm bảo các góc cùng phía nhận cùng một nhiễu
        # Tạo 4 nhiễu khác nhau cho 4 góc
        corner_noise = torch.zeros_like(vectors)
        for i in range(4):
            # Tạo nhiễu cho góc thứ i
            noise_i = rand_sign[:, i*2:i*2+2] * rand_part[:, i*2:i*2+2] * noise_scale[:, i*2:i*2+2]
            corner_noise[:, i*2:i*2+2] = noise_i
        
        # Áp dụng nhiễu vào các vector
        scaled_vectors = vectors * (1.0 + corner_noise)
        
        # Tính toán các điểm mới từ tâm và vector đã thêm nhiễu
        new_points = torch.zeros_like(known_bbox_poly)
        for i in range(0, 8, 2):
            new_points[:, i] = centers[:, 0] + scaled_vectors[:, i]
            new_points[:, i+1] = centers[:, 1] + scaled_vectors[:, i+1]

        known_bbox_poly = new_points.clamp(min=0.0, max=1.0)
        known_bbox_expand = poly2obb(known_bbox_poly)

    # Chuẩn bị embeddings
    label_enc = label_enc.to(device)
    m = known_labels_expaned.long().to(device)
    input_label_embed = label_enc(m)
    
    # Sử dụng inverse_sigmoid cho các tọa độ trong khoảng [0, 1]
    known_bbox_expand_normalized = known_bbox_expand.clone()
    known_bbox_expand_normalized[:, :4] = known_bbox_expand[:, :4].clamp(0.01, 0.99)  # Tránh các giá trị 0 hoặc 1
    input_bbox_embed = inverse_sigmoid(known_bbox_expand_normalized).to(device)
    
    # Padding
    padding_label = torch.zeros(pad_size, hidden_dim, device=device)
    padding_bbox = torch.zeros(pad_size, boxes.size(1), device=device)
    
    input_query_label = padding_label.repeat(batch_size, 1, 1)
    input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

    # Ánh xạ indices
    map_known_indice = torch.tensor([], device=device)
    if known_num:
        map_known_indice = torch.cat([torch.arange(num, device=device) for num in known_num])
        map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(2 * dn_number)]).long()
    
    if len(known_bid):
        input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
        input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed
    
    # Tạo attention mask với cải tiến để ngăn thông tin rò rỉ tốt hơn
    tgt_size = pad_size + num_queries
    attn_mask = torch.ones(tgt_size, tgt_size, device=device) < 0
    
    # Match query không thể thấy reconstruct
    attn_mask[pad_size:, :pad_size] = True
    
    # Reconstruct không thể thấy nhau
    for i in range(dn_number):
        start_idx = single_pad * 2 * i
        end_idx = single_pad * 2 * (i + 1)
        
        if i == 0:
            attn_mask[start_idx:end_idx, end_idx:pad_size] = True
        elif i == dn_number - 1:
            attn_mask[start_idx:end_idx, :start_idx] = True
        else:
            attn_mask[start_idx:end_idx, end_idx:pad_size] = True
            attn_mask[start_idx:end_idx, :start_idx] = True

    dn_meta = {
        "pad_size": pad_size,
        "num_dn_group": dn_number,
    }
    
    return input_query_label, input_query_bbox, attn_mask, dn_meta

def dn_post_process(outputs_class, outputs_coord, dn_meta, aux_loss, _set_aux_loss):
    """
    post process of dn after output from the transformer
    put the dn part in the dn_meta
    """
    if dn_meta and dn_meta["pad_size"] > 0:
        pad_size = dn_meta["pad_size"]
        
        # Tách phần DN và phần matching query
        output_known_class = outputs_class[:, :, :pad_size, :]
        output_known_coord = outputs_coord[:, :, :pad_size, :]
        
        # Cập nhật outputs để chỉ giữ lại phần matching query
        outputs_class = outputs_class[:, :, pad_size:, :]
        outputs_coord = outputs_coord[:, :, pad_size:, :]
        
        # Tạo dictionary chứa kết quả DN
        out = {
            "pred_logits": output_known_class[-1],
            "pred_boxes": output_known_coord[-1],
        }
        
        # Thêm auxiliary outputs nếu cần
        if aux_loss and callable(_set_aux_loss):
            out["aux_outputs"] = _set_aux_loss(output_known_class, output_known_coord)
        
        dn_meta["output_known_lbs_bboxes"] = out
        
    return outputs_class, outputs_coord