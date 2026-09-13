"""Thread-safe reservations prevent concurrent jobs from spending the same balance."""
import json
import time
import uuid

# Official standard rates retrieved 2026-09-12, per million tokens, <=272K input.
PRICES = {'gpt-5.6-sol': (4., .4, 20.), 'gpt-6-astra': (10., 1., 50.)}


class BudgetExceeded(RuntimeError):
    pass


def cost(model, input_tokens, cached_input_tokens, output_tokens):
    if min(input_tokens, cached_input_tokens, output_tokens) < 0 or cached_input_tokens > input_tokens:
        raise ValueError('Invalid token accounting')
    rates = PRICES[model]
    im, om = (2., 1.5) if input_tokens > 272000 else (1., 1.)
    return ((input_tokens-cached_input_tokens)*rates[0]*im +
            cached_input_tokens*rates[1]*im + output_tokens*rates[2]*om)/1e6


class Budget:
    def __init__(self, archive, limit):
        self.archive, self.limit = archive, float('inf') if limit is None else limit

    @property
    def spent(self):
        return self.archive.rows('SELECT COALESCE(SUM(usd),0) AS n FROM llm_calls')[0]['n']

    def reserve(self, role, model, generation, amount):
        with self.archive.lock:
            committed = self.archive.rows("SELECT COALESCE(SUM(CASE WHEN status='running' THEN reserved_usd ELSE usd END),0) AS n FROM llm_calls")[0]['n']
            if committed + amount > self.limit:
                raise BudgetExceeded(f'Budget reservation denied: {committed:.4f} + {amount:.4f} > {self.limit:.2f}')
            cid = uuid.uuid4().hex
            self.archive.put('llm_calls', id=cid, role=role, model=model, generation=generation,
                usd=0., reserved_usd=amount, status='running', created_at=time.time())
            return cid

    def finish(self, cid, model, usage=None, details=None):
        if usage is None:
            self.archive.execute("UPDATE llm_calls SET usd=reserved_usd,status='usage_unknown',details_json=? WHERE id=?",
                                 (json.dumps(details or {}), cid))
            return
        tokens = tuple(int(usage.get(k, 0)) for k in ['input_tokens', 'cached_input_tokens', 'output_tokens'])
        usd = cost(model, *tokens)
        self.archive.execute('UPDATE llm_calls SET input_tokens=?,cached_input_tokens=?,output_tokens=?,usd=?,status=?,details_json=? WHERE id=?',
                             (*tokens, usd, 'completed', json.dumps(details or {}, default=str), cid))
