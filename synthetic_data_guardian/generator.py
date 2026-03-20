"""Synthetic data generation using Faker."""
from faker import Faker
from typing import List, Dict, Any, Optional
import random

fake = Faker()


def generate_persons(n: int, seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """Generate n synthetic person records."""
    if seed is not None:
        Faker.seed(seed)
        random.seed(seed)
    return [
        {
            "id": i,
            "name": fake.name(),
            "email": fake.email(),
            "age": random.randint(18, 90),
            "salary": round(random.uniform(30000, 200000), 2),
            "city": fake.city(),
            "country": fake.country_code(),
        }
        for i in range(n)
    ]


def generate_transactions(n: int, seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """Generate n synthetic transaction records."""
    if seed is not None:
        Faker.seed(seed)
        random.seed(seed)
    return [
        {
            "id": i,
            "amount": round(random.uniform(1.0, 10000.0), 2),
            "currency": random.choice(["USD", "EUR", "GBP", "JPY"]),
            "timestamp": fake.iso8601(),
            "merchant": fake.company(),
            "category": random.choice(["food", "travel", "tech", "health", "retail"]),
        }
        for i in range(n)
    ]
