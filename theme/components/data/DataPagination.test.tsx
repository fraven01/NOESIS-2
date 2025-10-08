import * as React from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { axe } from "jest-axe";
import { DataPagination } from "./DataPagination";

describe("DataPagination", () => {
  it("navigates between pages", async () => {
    const user = userEvent.setup();

    const Example = () => {
      const [page, setPage] = React.useState(1);
      return <DataPagination current={page} total={3} onChange={setPage} />;
    };

    const { container } = render(<Example />);

    await user.click(screen.getByRole("button", { name: /next/i }));
    expect(screen.getByRole("button", { name: /prev/i })).not.toBeDisabled();

    await user.click(screen.getByRole("button", { name: /prev/i }));
    expect(screen.getByRole("button", { name: /prev/i })).toBeDisabled();

    expect(await axe(container)).toHaveNoViolations();
  });
});
