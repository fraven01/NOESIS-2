import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { axe } from "jest-axe";
import { Pagination } from "./Pagination";

describe("Pagination", () => {
  it("navigates pages and disables at ends", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const { container, rerender } = render(
      <Pagination current={1} total={3} onChange={onChange} />
    );
    const next = screen.getByRole("button", { name: /next/i });
    expect(next).not.toBeDisabled();
    await user.click(next);
    expect(onChange).toHaveBeenCalledWith(2);
    const prev = screen.getByRole("button", { name: /prev/i });
    expect(prev).toBeDisabled();
    const page2 = screen.getByRole("button", { name: "2" });
    await user.click(page2);
    expect(onChange).toHaveBeenCalledWith(2);
    rerender(<Pagination current={2} total={3} onChange={onChange} />);
    const prevEnabled = screen.getByRole("button", { name: /prev/i });
    await user.click(prevEnabled);
    expect(onChange).toHaveBeenCalledWith(1);
    const page3 = screen.getByRole("button", { name: "3" });
    await user.click(page3);
    expect(onChange).toHaveBeenCalledWith(3);
    rerender(<Pagination current={3} total={3} onChange={onChange} />);
    expect(screen.getByRole("button", { name: /next/i })).toBeDisabled();
    expect(await axe(container)).toHaveNoViolations();
  });
});
