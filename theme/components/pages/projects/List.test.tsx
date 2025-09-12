import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { axe } from "jest-axe";
import { List } from "./List";

const unsorted = [
  { id: "1", name: "Beta" },
  { id: "2", name: "Alpha" },
];
const data = [
  { id: "1", name: "Alpha" },
  { id: "2", name: "Beta" },
];

describe("Projects List", () => {
  it("sorts rows and sets aria-sort", async () => {
    render(<List projects={unsorted} />);
    const header = screen.getByRole("columnheader", { name: /name/i });
    expect(header).toHaveAttribute("aria-sort", "none");
    await userEvent.click(header);
    expect(header).toHaveAttribute("aria-sort", "ascending");
    let rows = screen.getAllByRole("row");
    expect(rows[1]).toHaveTextContent("Alpha");
    await userEvent.click(header);
    expect(header).toHaveAttribute("aria-sort", "descending");
    rows = screen.getAllByRole("row");
    expect(rows[1]).toHaveTextContent("Beta");
  });

  it("handles selection including mixed state", async () => {
    render(<List projects={data} />);
    const selectAll = screen.getByLabelText("Select all");
    const first = screen.getByLabelText("Select Alpha");
    await userEvent.click(first);
    expect(selectAll).toHaveAttribute("aria-checked", "mixed");
    await userEvent.click(selectAll);
    expect(selectAll).toHaveAttribute("aria-checked", "true");
    expect(first).toHaveAttribute("aria-checked", "true");
  });

  it("has logical keyboard navigation", async () => {
    render(<List projects={data} />);
    await userEvent.tab();
    expect(screen.getByLabelText("Search")).toHaveFocus();
    await userEvent.tab();
    expect(screen.getByLabelText("Filter")).toHaveFocus();
    await userEvent.tab();
    expect(screen.getByLabelText("Select all")).toHaveFocus();
    await userEvent.tab();
    expect(screen.getByLabelText("Select Alpha")).toHaveFocus();
  });

  it("is accessible", async () => {
    const { container } = render(<List projects={data} />);
    expect(await axe(container)).toHaveNoViolations();
  });

  it("shows error state with alert role", async () => {
    const { container } = render(<List projects={data} status="error" />);
    expect(screen.getByRole("alert")).toHaveTextContent(
      "Something went wrong"
    );
    expect(await axe(container)).toHaveNoViolations();
  });
});
