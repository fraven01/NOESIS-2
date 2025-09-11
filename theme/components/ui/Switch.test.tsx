import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { axe } from "jest-axe";
import { Switch } from "./Switch";

describe("Switch", () => {
  it("toggles and calls onCheckedChange", async () => {
    const user = userEvent.setup();
    const onCheckedChange = vi.fn();
    const { container, rerender } = render(
      <Switch
        aria-label="toggle"
        checked={false}
        onCheckedChange={onCheckedChange}
      />
    );
    const sw = screen.getByRole("switch");
    expect(sw).toHaveAttribute("aria-checked", "false");
    await user.click(sw);
    expect(onCheckedChange).toHaveBeenCalledWith(true);
    rerender(
      <Switch
        aria-label="toggle"
        checked={true}
        onCheckedChange={onCheckedChange}
      />
    );
    const swChecked = screen.getByRole("switch");
    expect(swChecked).toHaveAttribute("aria-checked", "true");
    expect(await axe(container)).toHaveNoViolations();
  });
});
