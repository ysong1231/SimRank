import sys

BAR_LENGTH = 30

def update_progress(progress):
    STATUS = ''
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        raise ValueError('Progress must be float')
    if progress < 0:
        raise ValueError('Progress below 0')
    if progress >= 1:
        progress = 1
        STATUS = 'Done...\r\n'
    block = int(round(BAR_LENGTH * progress))
    text = f'\rPercent: [{"#" * block + "-" * (BAR_LENGTH - block)}] {round(progress * 100, 1)}% {STATUS}'
    sys.stdout.write(text)
    sys.stdout.flush()