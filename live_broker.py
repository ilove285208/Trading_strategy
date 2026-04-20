import importlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class BrokerPosition:
    quantity: float
    metadata: dict | None = None


@dataclass
class OrderRequest:
    symbol: str
    target_quantity: float
    current_quantity: float
    delta_quantity: float
    side: str
    order_type: str
    time_in_force: str
    signal_payload: dict


@dataclass
class OrderResult:
    status: str
    order_id: str | None = None
    message: str | None = None
    payload: dict | None = None


class BaseBrokerAdapter(ABC):
    def connect(self):
        return None

    def disconnect(self):
        return None

    @abstractmethod
    def get_position(self, symbol):
        raise NotImplementedError

    @abstractmethod
    def submit_target_order(self, order_request):
        raise NotImplementedError


class DryRunBrokerAdapter(BaseBrokerAdapter):
    def __init__(self, current_quantity=0.0):
        self.current_quantity = float(current_quantity)

    def get_position(self, symbol):
        return BrokerPosition(quantity=self.current_quantity, metadata={"symbol": symbol})

    def submit_target_order(self, order_request):
        self.current_quantity = float(order_request.target_quantity)
        return OrderResult(
            status="simulated",
            order_id=f"dryrun-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            message="Dry-run adapter simulated the target-position order.",
            payload={"symbol": order_request.symbol},
        )


class FileBrokerAdapter(BaseBrokerAdapter):
    def __init__(self, order_outbox, broker_state_path=None):
        self.order_outbox = Path(order_outbox)
        self.order_outbox.mkdir(parents=True, exist_ok=True)
        self.broker_state_path = (
            Path(broker_state_path) if broker_state_path is not None else None
        )

    def _read_state_payload(self):
        if self.broker_state_path is None or not self.broker_state_path.exists():
            return {}
        with self.broker_state_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def get_position(self, symbol):
        payload = self._read_state_payload()
        positions = payload.get("positions", {})
        if symbol in positions:
            symbol_position = positions[symbol]
            if isinstance(symbol_position, dict):
                quantity = float(symbol_position.get("quantity", 0.0))
            else:
                quantity = float(symbol_position)
        else:
            quantity = float(payload.get("quantity", 0.0))
        return BrokerPosition(quantity=quantity, metadata=payload)

    def submit_target_order(self, order_request):
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        order_id = f"{order_request.symbol}-{timestamp}"
        order_payload = {
            "order_id": order_id,
            "created_at_utc": timestamp,
            "symbol": order_request.symbol,
            "target_quantity": order_request.target_quantity,
            "current_quantity": order_request.current_quantity,
            "delta_quantity": order_request.delta_quantity,
            "side": order_request.side,
            "order_type": order_request.order_type,
            "time_in_force": order_request.time_in_force,
            "signal": order_request.signal_payload,
        }
        output_path = self.order_outbox / f"{order_id}.json"
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(order_payload, file, ensure_ascii=False, indent=2)
        return OrderResult(
            status="queued",
            order_id=order_id,
            message=f"Order instruction written to {output_path}",
            payload={"output_path": str(output_path)},
        )


def load_custom_adapter(spec, args):
    if ":" not in spec:
        raise ValueError("Custom adapter spec must look like 'module.path:ClassName'.")
    module_name, class_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    adapter_class = getattr(module, class_name)
    return adapter_class(args)


def create_broker_adapter(args):
    if args.broker == "dry-run":
        return DryRunBrokerAdapter(current_quantity=args.current_quantity)
    if args.broker == "file":
        return FileBrokerAdapter(
            order_outbox=args.order_outbox,
            broker_state_path=args.broker_state_path,
        )
    if args.broker == "custom":
        if not args.custom_adapter:
            raise ValueError("--custom-adapter is required when --broker custom is used.")
        return load_custom_adapter(args.custom_adapter, args)
    raise ValueError(f"Unsupported broker adapter: {args.broker}")
