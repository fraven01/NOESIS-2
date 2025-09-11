import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { axe } from "jest-axe";
import { Checkbox } from "./Checkbox";

describe("Checkbox", () => {
  it("toggles checked state on click", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const { container } = render(
      <Checkbox aria-label="accept" onChange={onChange} />
    );
    const cb = screen.getByRole("checkbox");
    expect(cb).not.toBeChecked();
    await user.click(cb);
    expect(cb).toBeChecked();
    await user.click(cb);
    expect(cb).not.toBeChecked();
    expect(onChange).toHaveBeenCalledTimes(2);
    expect(await axe(container)).toHaveNoViolations();
  });
});
