def universal_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Universal collate function for both EVA and CLIP denoising"""
    if not batch:
        raise ValueError("Empty batch")
    
    # Filter valid items
    valid_batch = [item for item in batch if item is not None]
    if not valid_batch:
        raise ValueError("No valid items in batch")
    
    # Get first item to determine structure
    first_item = valid_batch[0]
    task_mode = first_item.get('task_mode', 'unknown')
    
    # Stack tensors
    collated = {}
    
    # Main tensors
    tensor_keys = ['input_embeddings', 'conditioning_embeddings', 'target_embeddings', 
                   'noise', 'noise_level']
    
    for key in tensor_keys:
        if key in first_item:
            if key == 'noise_level':
                # Scalar values
                values = torch.tensor([item[key] for item in valid_batch])
            else:
                # Tensor values
                values = torch.stack([item[key] for item in valid_batch])
            collated[key] = values
    
    # Create flow matching inputs
    if 'input_embeddings' in collated and 'noise_level' in collated:
        # These are the noisy embeddings at timestep t
        collated['hidden_states'] = collated['input_embeddings']
        collated['timestep'] = collated['noise_level']
        collated['encoder_hidden_states'] = collated['conditioning_embeddings']
    
    # Metadata
    collated['batch_size'] = len(valid_batch)
    collated['task_mode'] = task_mode
    
    # Lists
    if 'caption' in first_item:
        collated['captions'] = [item['caption'] for item in valid_batch]
    if 'key' in first_item:
        collated['keys'] = [item['key'] for item in valid_batch]
    
    # Dimensions info
    for key in ['input_dim', 'output_dim', 'conditioning_dim', 'num_tokens']:
        if key in first_item:
            collated[key] = first_item[key]
    
    return collated