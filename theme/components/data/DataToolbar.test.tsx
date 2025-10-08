import * as React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { axe } from "jest-axe";
import { vi } from "vitest";
import { DataToolbar } from "./DataToolbar";

describe("DataToolbar", () => {
  it("updates search value", async () => {
    const user = userEvent.setup();
    const onSearch = vi.fn();

    const Example = () => {
      const [search, setSearch] = React.useState("");
      return (
        <DataToolbar
          search={search}
          onSearch={(value) => {
            setSearch(value);
            onSearch(value);
          }}
          selectedCount={0}
          onDeleteSelected={() => {}}
        />
      );
    };

    const { container } = render(<Example />);
    const input = screen.getByRole("textbox", { name: /search/i });

    await user.type(input, "alpha");

    await waitFor(() => expect(input).toHaveValue("alpha"));
    expect(onSearch).toHaveBeenCalled();
    expect(await axe(container)).toHaveNoViolations();
  });

  it("shows delete button when items selected", async () => {
    const user = userEvent.setup();
    const onDelete = vi.fn();
    const { container } = render(
      <DataToolbar search="" onSearch={() => {}} selectedCount={2} onDeleteSelected={onDelete} />
    );

    const button = screen.getByRole("button", { name: /delete selected/i });
    await user.click(button);
    expect(onDelete).toHaveBeenCalled();
    expect(await axe(container)).toHaveNoViolations();
  });
});
