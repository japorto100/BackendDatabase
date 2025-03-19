import torch
import logging

logger = logging.getLogger(__name__)

def process_qwen_response(model, processor, device, messages):
    """Process multimodal messages with Qwen model"""
    try:
        # Extract images and process
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Process images from messages
        image_inputs = []
        for message in messages:
            for content in message["content"]:
                if content.get("type") == "image":
                    try:
                        from PIL import Image
                        image = Image.open(content["image"])
                        image_inputs.append(image)
                    except Exception as e:
                        logger.error(f"Error processing image: {e}")
                        
        # Process inputs
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)
        
        # Generate response
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # Decode response
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text[0]
    except Exception as e:
        logger.error(f"Error processing Qwen response: {e}")
        return f"Error generating response: {str(e)}"
