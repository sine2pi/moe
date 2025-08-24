Different gates for gating.. They were written with multiple audio features in mind but are pretty generic. I'll continue to add more as I organize my stuff.

## Gates 

``` python


class cgate(nn.Module):
    def __init__(self, dims, enabled=True):
        super().__init__()
        self.enabled = enabled
        if enabled:
            self.s_gate = nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid())
            self.w_gate = nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid())
            self.p_gate = nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid())
            self.e_gate = nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid())
            self.ph_gate = nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid())
            self.integ = nn.Linear(dims*5, dims)
            self.reset_parameters()
        
    def forward(self, x, enc):
        if not self.enabled:
            return None
        s_feat = enc.get("spectrogram", x)
        w_feat = enc.get("waveform", x)
        p_feat = enc.get("pitch", x)
        e_feat = enc.get("envelope", x)
        ph_feat = enc.get("phase", x)
        s = self.s_gate(x) * s_feat
        w = self.w_gate(x) * w_feat
        p = self.p_gate(x) * p_feat
        e = self.e_gate(x) * e_feat
        ph = self.ph_gate(x) * ph_feat
        comb = torch.cat([s, w, p, e, ph], dim=-1)
        return self.integ(comb)

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

class mgate(nn.Module):
    def __init__(self, dims, mem=64):
        super().__init__()
        self.mk = nn.Parameter(torch.randn(mem, dims))
        self.mv = nn.Parameter(torch.randn(mem, 1))
        self.mg = nn.Sequential(nn.Linear(dims, dims//2), nn.SiLU(), nn.Linear(dims//2, 1))
        self.reset_parameters()

    def forward(self, x, cos=False):
        if cos:
            key = F.softmax(torch.matmul(F.normalize(x, p=2, dim=-1), F.normalize(self.mk, p=2, dim=-1).transpose(0, 1)) / math.sqrt(x.shape[-1]), dim=-1)
        else:
            key = F.softmax(torch.matmul(x, self.mk.transpose(0, 1)) / math.sqrt(x.shape[-1]), dim=-1)
        return 0.5 * (torch.sigmoid(self.mg(x)) + torch.sigmoid(torch.matmul(key, self.mv)))

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

class tmgate(nn.Module):
    def __init__(self, dims: int, head: int, memory_size: int = 64, threshold: float = 0.8):
        super().__init__()
        self.dims=dims
        self.mkeys = {}

        self.xa_proj = nn.Linear(dims, dims)
        self.activation = nn.Sigmoid()
        self.pattern = lambda length: sinusoids(length, dims=dims)  
        self.primer = torch.ones(1, 512, dims)
        self.reset_parameters()

    def forward(self, x) -> torch.Tensor: 
        if x is None:
            cur = self.primer
            self.key = cur
        else:
            cur = self.pattern(x.shape[1]).to(device, dtype)
    
        self.mkeys["last"] = self.key
        cur = self.xa_proj(cur.mean(dim=1)) 

        for b in range(cur.size(0)):
            cur_xa = cur[b]
            score = -1.0
            best = None
            for last in self.mkeys.items():
                last = self.mkeys["last"]
                similarity = F.cosine_similarity(cur_xa, last, dim=0).mean()

                if similarity > score:
                    score = similarity
                    best = best

            gating_value = self.activation(torch.tensor(score))
            if gating_value > self.threshold and best is not None:
                self.key = cur
            else:
                self.key = last
            threshold = apply_ste_threshold(x, self.threshold)
        return threshold, self.key

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

class StraightThroughThreshold(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold):
        binary_output = (x > threshold).float()
        ctx.save_for_backward(x, threshold)
        return binary_output

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors
        grad_x = grad_output.clone()
        grad_threshold = None
        return grad_x, grad_threshold

apply_ste_threshold = StraightThroughThreshold.apply

class LinearGate(nn.Linear):
    def __init__(self, in_features, out_features, act="swish", norm_type=None, context = 4, num_types=4, top=4):
        super().__init__(in_features, out_features, bias=False, device=device, dtype=dtype)
        self.context = context
        self.top = top
        self.num_types = num_types
        self.act=act

        self.gate = nn.Sequential(get_norm(norm_type, in_features), nn.Linear(in_features, out_features, bias=False), nn.Softmax(dim=-1))
        self.context = nn.Parameter(torch.ones(context), requires_grad=True)
        self.bias2 = nn.Parameter(torch.zeros(context, in_features), requires_grad=True)
        self.reset_parameters()

    def forward(self, x, top=4):
        x = x.unsqueeze(-2)
        x = x * (1 + get_activation(self.act)((rearrange(self.context, 'c -> 1 1 c 1') * self.gate(x).squeeze(-1)).mean(-1, keepdim=True)))
        x = x + self.bias2.unsqueeze(0)
        _, indices = torch.topk(x, self.top, dim=-2, sorted=False)
        x = torch.gather(x, -2, indices).mean(dim=-2)
        x = super().forward(x)
        return x

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

class FeatureGate(nn.Module):
    def __init__(self, dims: int, expand: int, adapt_dim: int):
        super().__init__()

        self.steps = nn.ModuleList([LinearGate(dims, dims, adapt_dim) for _ in range(expand)])
        self.gating = nn.Linear(adapt_dim, expand)
        self.reset_parameters()

    def forward(self, xa, xb): # for audio features 
        scores = F.softmax(self.gating(xa), dim=-1)
        output = sum(scores[:, i].unsqueeze(1) * gate(xa, xb) for i, gate in enumerate(self.steps))
        return output

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

class dgate(nn.Module):
    def __init__(self, dims: int, head: int, threshold: float = 0.8):
        super().__init__()
        self.dims = dims
        
        self.register_buffer('last_key', torch.zeros(1, dims))
        self.xa_proj = nn.Linear(dims, dims)
        self.activation = nn.Sigmoid()
        self.register_buffer('primer', torch.ones(1, 512, dims))
        self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32), requires_grad=True)
        self.reset_parameters()

    def forward(self, x: torch.Tensor): 
        device, dtype = x.device, x.dtype if x is not None else self.primer.device, self.primer.dtype
        if x is None:
            cur_representation = self.primer.to(device, dtype).mean(dim=1)
        else:
            pattern_tensor = self.pattern(x.shape[1]).to(device, dtype)
            cur_representation = pattern_tensor.mean(dim=1)

        cur_proj = self.xa_proj(cur_representation)
        expanded_last_key = self.last_key.expand(cur_proj.shape[0], -1) 
        similarity_scores = F.cosine_similarity(cur_proj, expanded_last_key, dim=-1).unsqueeze(-1)
        gating_value = self.activation(similarity_scores)
        decision = apply_ste_threshold(gating_value, self.threshold)
        self.last_key.copy_(cur_representation.detach()) 
        return decision, gating_value

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

class m_gate(nn.Module):
    def __init__(self, dims, mem_size=64):
        super().__init__()

        self.m_key = nn.Parameter(torch.randn(mem_size, dims))
        self.m_val = nn.Parameter(torch.randn(mem_size, 1))
        self.gate_proj = nn.Sequential(nn.Linear(dims, dims//2), nn.SiLU(), nn.Linear(dims//2, 1))
        self.reset_parameters()
            
    def forward(self, x):
        d_gate = torch.sigmoid(self.gate_proj(x))
        attention = torch.matmul(x, self.m_key.transpose(0, 1))
        attention = F.softmax(attention / math.sqrt(x.shape[-1]), dim=-1)
        m_gate = torch.matmul(attention, self.m_val)
        m_gate = torch.sigmoid(m_gate)
        return 0.5 * (d_gate + m_gate)

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

class lgate(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid())
        self.reset_parameters()

    def forward(self, x):
        return self.gate(x)

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

class tgate(nn.Module):
    def __init__(self, dims, num_types=2):
        super().__init__()

        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid()) for _ in range(num_types)])
        self.classifier = nn.Sequential(nn.Linear(dims, num_types), nn.Softmax(dim=-1))
        self.reset_parameters()

    def forward(self, x):
        types = self.classifier(x)
        gates = torch.stack([gate(x) for gate in self.gates], dim=-1)
        return  torch.sum(gates * types.unsqueeze(2), dim=-1)

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

class tgate_hybrid(nn.Module):
    def __init__(self, dims, num_types=10, k=2):
        super().__init__()
        self.num_types = num_types
        self.k = k
        
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid()) for _ in range(num_types)])
        self.classifier = nn.Sequential(nn.Linear(dims, num_types), nn.Softmax(dim=-1))
        self.sparse_classifier = nn.Linear(dims, num_types)
        self.alpha = nn.Parameter(torch.ones(1))
        self.reset_parameters()

    def forward(self, x):
        soft_types = self.classifier(x)
        logits = self.sparse_classifier(x)
        top_k_logits, top_k_indices = torch.topk(logits, self.k, dim=-1)
        sparse_gates_values = F.softmax(top_k_logits, dim=-1)
        sparse_types = torch.zeros_like(soft_types)
        sparse_types.scatter_(-1, top_k_indices, sparse_gates_values)
        gates = torch.stack([gate(x) for gate in self.gates], dim=-1)
        mixed_types = torch.sigmoid(self.alpha) * sparse_types + (1 - torch.sigmoid(self.alpha)) * soft_types
        return torch.sum(gates * mixed_types.unsqueeze(2), dim=-1)
    
    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

class tgate_conditional(nn.Module):
    def __init__(self, dims, num_types=2, k=1, use_sparse=False):
        super().__init__()
        self.num_types = num_types
        self.k = k
        self.use_sparse = use_sparse
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid()) for _ in range(num_types)])
        
        if self.use_sparse:
            self.classifier = nn.Linear(dims, num_types)
        else:
            self.classifier = nn.Sequential(nn.Linear(dims, num_types), nn.Softmax(dim=-1))
        self.reset_parameters()

    def forward(self, x):
        if self.use_sparse:
            logits = self.classifier(x)
            top_k_logits, top_k_indices = torch.topk(logits, self.k, dim=-1)
            sparse_gates_values = F.softmax(top_k_logits, dim=-1)
            types = torch.zeros_like(logits)
            types.scatter_(-1, top_k_indices, sparse_gates_values)
        else:
            types = self.classifier(x)
        gates = torch.stack([gate(x) for gate in self.gates], dim=-1)
        return torch.sum(gates * types.unsqueeze(2), dim=-1)

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

class tgate_topk(nn.Module):
    def __init__(self, dims, num_experts=4, k=2):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.classifier = nn.Linear(dims, num_experts)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid()) for _ in range(num_experts)])
        self.reset_parameters()

    def forward(self, x):
        logits = self.classifier(x)
        top_k_logits, top_k_indices = torch.topk(logits, self.k, dim=-1)
        mask = torch.zeros_like(logits, requires_grad=False)
        mask.scatter_(-1, top_k_indices, 1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)
        gates = torch.zeros_like(logits)
        gates.scatter_(-1, top_k_indices, top_k_gates)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        return torch.sum(expert_outputs * gates.unsqueeze(2), dim=-1)

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

class BiasingGateRefactored(nn.Module):
    def __init__(self, dims: int, head: int, memory_size: int = 64, threshold: float = 0.8):
        super(BiasingGateRefactored, self).__init__()
        self.memory_size = memory_size
        self.threshold = threshold
        
        self.register_buffer("mkeys_pattern", sinusoids(memory_size, dims))
        self.mkeys_bias = nn.Parameter(torch.randn(memory_size, head))
        
        self.xa_projection = nn.Linear(dims, dims)
        self.activation = nn.Sigmoid()
        
    def forward(self, x, xa) -> torch.Tensor:
        if x is None:
            return None
        
        xa_projected = self.xa_projection(xa.mean(dim=1))
        similarity = F.cosine_similarity(xa_projected.unsqueeze(1), self.mkeys_pattern.unsqueeze(0), dim=-1)
        
        best_scores, best_indices = torch.max(similarity, dim=1)
        best_bias_weights = self.mkeys_bias[best_indices]
        gating_value = self.activation(best_scores)
        
        mask = (gating_value > self.threshold).float().unsqueeze(-1)
        
        shot_bias = x
        scaled_bias = shot_bias * best_bias_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        final_bias = mask * scaled_bias + (1 - mask) * shot_bias
        return final_bias

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

class SimpleMoE(nn.Module):
    def __init__(self, num_experts: int, expert_size: int, output_size: int):
        super().__init__()

        self.experts = nn.ModuleList([nn.Linear(expert_size, output_size) for _ in range(num_experts)])
        self.gate = nn.Linear(expert_size, num_experts)

    def forward(self, x: Tensor) -> Tensor:
  
        gate_outputs = F.softmax(self.gate(x), dim=1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        return torch.einsum("ab,abc->ac", gate_outputs, expert_outputs)

class LessSimpleMoE(nn.Module):
    def __init__(self, dims, num_experts=5, num_modalities=5, k=1, memory_size=64, threshold=0.8):
        super().__init__()
        self.dims = dims
        self.num_experts = num_experts
        self.num_modalities = num_modalities
        self.k = k
        self.memory_size = memory_size
        self.threshold = threshold
        
        self.soft_router = nn.Sequential(nn.Linear(dims, num_experts), nn.Softmax(dim=-1))
        self.sparse_router = nn.Linear(dims, num_experts)

        self.m_key = nn.Parameter(torch.randn(memory_size, dims))
        self.m_val = nn.Parameter(torch.randn(memory_size, num_experts))
        self.direct_gate = nn.Sequential(nn.Linear(dims, num_experts), nn.Sigmoid())

        self.experts = nn.ModuleList( [nn.Sequential(nn.Linear(dims, dims), nn.SiLU(), nn.Linear(dims, dims)) for _ in range(num_experts)] )
        
        self.fusion_gates = nn.ModuleList( [nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid()) for _ in range(num_modalities)] )
        self.fusion_projection = nn.Linear(dims * num_modalities, dims)
        
        self.register_buffer("alpha_moe", torch.tensor(0.5))
        self.register_buffer("alpha_fusion", torch.tensor(0.5))
        self.reset_parameters()
    
    def forward(self, x, enc):

        soft_expert_weights = self.soft_router(x)
        
        logits = self.sparse_router(x)
        top_k_logits, top_k_indices = torch.topk(logits, self.k, dim=-1)
        sparse_gates = F.softmax(top_k_logits, dim=-1)
        
        sparse_expert_weights = torch.zeros_like(logits)
        sparse_expert_weights.scatter_(-1, top_k_indices, sparse_gates)
        
        attention = torch.matmul(x, self.m_key.transpose(0, 1))
        attention = F.softmax(attention / math.sqrt(self.dims), dim=-1)

        memory_weights = torch.matmul(attention, self.m_val)
        memory_weights = F.softmax(memory_weights, dim=-1)

        mixed_expert_weights = (self.alpha_moe * sparse_expert_weights + (1 - self.alpha_moe) * soft_expert_weights)
        final_expert_weights = (mixed_expert_weights + memory_weights) / 2
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        moe_output = torch.sum(expert_outputs * final_expert_weights.unsqueeze(1), dim=-1)

        features = [enc.get("spectrogram", x), enc.get("waveform", x), enc.get("pitch", x), enc.get("envelope", x), enc.get("phase", x)]
        gates = [gate(x) for gate in self.fusion_gates]
        
        scaled_features = [gates[i] * features[i] for i in range(len(features))]
        fused_output = self.fusion_projection(torch.cat(scaled_features, dim=-1))
        
        return (self.alpha_fusion * fused_output) + ((1 - self.alpha_fusion) * moe_output)

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

class BiasingGateB(nn.Module):
    def __init__(self, dims: int, head: int, memory_size: int = 64, threshold: float = 0.8):
        super().__init__()
        self.dims = dims
        self.head = head
        self.memory_size = memory_size
        self.threshold = threshold
        self.mkeys = {}
        self.p = nn.Linear(dims, dims)
        self.tgate = nn.Sigmoid()

        self.pattern = lambda length, dims, max_tscale: sinusoids(length, dims)  
        self.one_shot = OneShot(dims, head)
        self.reset_parameters()

        for _ in range(memory_size): # example
            pattern = lambda length, dims, max_tscale: sinusoids(length, dims) 
            bias_weight = OneShot(dims, head)
            self.mkeys[tuple(pattern.tolist())] = bias_weight

    def forward(self, x, xa) -> torch.Tensor:
        B, T, _ = x.shape
        input = self.p(x.mean(dim=1))
        batch_gate_biases = []
        for b in range(B):
            cur_input = input[b]
            score = -1.0
            best_match = None
            for pattern, gate_bias in self.mkeys.items():
                pattern_tensor = torch.tensor(pattern).to(cur_input.device)
                similarity = F.cosine_similarity(cur_input, pattern_tensor, dim=0)
                if similarity > score:
                    score = similarity
                    best_match = gate_bias
            gating_value = self.tgate(score.unsqueeze(0))
            if gating_value > self.threshold and best_match is not None:
                batch_gate_biases.append(best_match.unsqueeze(0))
            else:
                batch_gate_biases.append(torch.zeros(1, self.head).to(x.device))
        return torch.cat(batch_gate_biases, dim=0)

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

class m2gate(nn.Module):
    def __init__(self, dims, mem=64, thresh=0.5):
        super().__init__()

        self.mkeys = nn.ParameterList([
            nn.Parameter(torch.randn(dims)),
            nn.Parameter(torch.randn(dims)),
        ])
        
        self.key_matrix = nn.Parameter(torch.randn(mem, dims))
        self.val_matrix = nn.Parameter(torch.randn(mem, 1))
        self.mlp = nn.Sequential(nn.Linear(dims, dims // 2), nn.SiLU(), nn.Linear(dims // 2, 1))
        self.threshold = nn.Parameter(torch.tensor(thresh, dtype=torch.float32), requires_grad=False).to(device)
        self.concat_layer = nn.Linear(2, 1, device=device, dtype=dtype)
        self.tgate_activation = nn.Sigmoid() 
        self.xa_projection = nn.Linear(dims, dims)

        self.register_buffer('previous_best_pattern', None)

    def forward(self, x):
        x_processed = self.xa_projection(x.mean(dim=1))
        skip_indicators = torch.ones(x_processed.size(0), dtype=torch.float32, device=x.device)

        previous_input_pattern_in_batch = None 
        for b in range(x_processed.size(0)):
            cur_x_element = x_processed[b]
            score = -1.0
            current_best_pattern = None 

            for pattern_tensor in self.mkeys:
                similarity = F.cosine_similarity(cur_x_element, pattern_tensor, dim=0)

                if similarity > score:
                    score = similarity
                    current_best_pattern = pattern_tensor

            if previous_input_pattern_in_batch is None or not torch.equal(current_best_pattern, previous_input_pattern_in_batch):
                skip_indicators[b] = 1.0
            else:
                skip_indicators[b] = 0.0
            previous_input_pattern_in_batch = current_best_pattern 
        change_scores = torch.zeros(x_processed.size(0), dtype=torch.float32, device=x.device)

        previous_input_pattern_in_batch = None 
        
        for b in range(x_processed.size(0)):
            cur_x_element = x_processed[b]
            score = -1.0
            current_best_pattern = None

            for pattern_tensor in self.mkeys:
                similarity = F.cosine_similarity(cur_x_element, pattern_tensor, dim=0)
                if similarity > score:
                    score = similarity
                    current_best_pattern = pattern_tensor

            if previous_input_pattern_in_batch is None or not torch.equal(current_best_pattern, previous_input_pattern_in_batch):
                change_scores[b] = 1.0
            else:
                change_scores[b] = 0.0
            previous_input_pattern_in_batch = current_best_pattern

        scalar = apply_ste_threshold(change_scores.unsqueeze(-1), self.threshold)
        key = F.softmax(torch.matmul(F.normalize(x_processed, p=2, dim=-1), F.normalize(self.key_matrix, p=2, dim=-1).transpose(0, 1)) / math.sqrt(x_processed.shape[-1]), dim=-1)
        gate = self.concat_layer(torch.cat((torch.matmul(key, self.val_matrix),  self.mlp(x_processed)), dim=-1))
        return scalar, gate

class Memory(nn.Module):
    def __init__( self, new_dims: int, old_dims: int):
        super().__init__()

        self.new_dims = new_dims
        self.old_dims = old_dims
        self.new = nn.Linear(new_dims, old_dims)
        self.old = nn.Linear(old_dims, old_dims)
        self.activation = nn.Tanh()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        mix = self.activation(self.new(x) + self.old(y))
        return mix

    def initialize_old(self, batch: int, device: torch.device) -> Tensor:
        old_dims = torch.zeros(batch, self.old_dims).to(device)
        return old_dims

class MixtureOfMemories(nn.Module):
    def __init__(self, dims, head, num_experts: int, expert_size: int, output_size: int):
        super().__init__()
        self.mems = nn.ModuleList([nn.Linear(expert_size, output_size) for _ in range(num_experts)])
        self.gate = nn.Linear(expert_size, num_experts)
        self.attention = nn.MultiheadAttention(dims, head)
        self.memory = Memory(dims, dims)
        self.layernorm = nn.LayerNorm(dims)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        output, _ = self.attention(x, x, x)
        y = self.memory(output.mean(dim=0), y)
        gate_outputs = F.softmax(self.gate(x), dim=1)
        _outputs = torch.stack([mem(x) for mem in self.mems], dim=1)
        _output = torch.einsum("ab,abc->ac", gate_outputs, _outputs)
        output = self.layernorm(output + _output.unsqueeze(0))
        return output

class Transformer(nn.Module):
    def __init__( self, dims: int, head: int, num_experts: int, expert_size: int, layer: int):
        super().__init__()

        self.layers = nn.ModuleList([MixtureOfMemories( dims, head, num_experts, expert_size ) for _ in range(layer)])
        self.y = torch.zeros(1, dims)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            return layer(x, self.y)
      
