import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { axe } from "jest-axe";
import { DataPagination } from "./DataPagination";

describe("DataPagination", () => {
  it("proxies pagination interactions and is accessible", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const { container } = render(
      <DataPagination current={2} total={4} onChange={onChange} />
    );

    const next = screen.getByRole("button", { name: /next/i });
    await user.click(next);
    expect(onChange).toHaveBeenNthCalledWith(1, 3);

    const prev = screen.getByRole("button", { name: /prev/i });
    await user.click(prev);
    expect(onChange).toHaveBeenNthCalledWith(2, 1);

    const firstPage = screen.getByRole("button", { name: "1" });
    await user.click(firstPage);
    expect(onChange).toHaveBeenNthCalledWith(3, 1);

    expect(await axe(container)).toHaveNoViolations();
  });
});
