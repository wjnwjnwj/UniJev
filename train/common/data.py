"""Record modality from its state's media items."""


def modality(record):
    state = record['request']['state']
    kinds = set()
    if isinstance(state, dict):
        for message in state['messages']:
            if isinstance(message['content'], list):
                kinds.update(item['type'] for item in message['content'] if item['type'] in {'image_url', 'video_url'})
    return 'mixed' if len(kinds) > 1 else 'image' if 'image_url' in kinds else 'video' if 'video_url' in kinds else 'text'
