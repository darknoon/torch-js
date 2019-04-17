import * as torch from "../../lib";
import { tensor, Tensor } from "../../lib";

test("can add two Float32 tensors", () => {
  const a = tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);
  const b = tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]);

  const c = torch.add(a, b);

  const expected = tensor([[1.1, 1.2, 1.3], [1.4, 1.5, 1.6]]);

  expect(c.toObject().data).toEqual(expected.toObject().data);
});

test("can add Float32 tensor and Number", () => {
  const a = tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);
  const b = 1;

  const c = torch.add(a, b);
  const expected = tensor([[1.1, 1.2, 1.3], [1.4, 1.5, 1.6]]);

  expect(c.toObject().data).toEqual(expected.toObject().data);
});
