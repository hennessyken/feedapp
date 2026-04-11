from subscribers.base import BaseSubscriber, SubscriberContext


def TelegramSubscriber(**kwargs):
    from subscribers.telegram import TelegramSubscriber as _cls
    return _cls(**kwargs)


def TraderSubscriber(**kwargs):
    from subscribers.trader import TraderSubscriber as _cls
    return _cls(**kwargs)


__all__ = [
    "BaseSubscriber",
    "SubscriberContext",
    "TelegramSubscriber",
    "TraderSubscriber",
]
