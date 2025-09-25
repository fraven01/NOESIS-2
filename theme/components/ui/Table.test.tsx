import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { axe } from "jest-axe";
import { Table, TableHead, TableRow, SortableHeader, TableBody, TableCell } from "./Table";

describe("Table", () => {
  const setup = () => (
    <Table caption="Users">
      <TableHead>
        <TableRow>
          <SortableHeader id="name">Name</SortableHeader>
        </TableRow>
      </TableHead>
      <TableBody>
        <TableRow>
          <TableCell>Alice</TableCell>
        </TableRow>
      </TableBody>
    </Table>
  );

  it("renders caption and is accessible", async () => {
    const { container } = render(setup());
    expect(screen.getByText("Users")).toBeInTheDocument();
    expect(await axe(container)).toHaveNoViolations();
  });

  it("toggles aria-sort on header click", async () => {
    render(setup());
    const header = screen.getByRole("columnheader", { name: /name/i });
    expect(header).not.toHaveAttribute("aria-sort");
    await userEvent.click(header);
    expect(header).toHaveAttribute("aria-sort", "asc");
    await userEvent.click(header);
    expect(header).toHaveAttribute("aria-sort", "desc");
  });
});
