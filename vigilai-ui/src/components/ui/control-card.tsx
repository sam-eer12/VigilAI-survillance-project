import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface ControlCardProps {
  title: string;
  children: React.ReactNode;
  className?: string;
  action?: React.ReactNode;
}

export function ControlCard({ title, children, className, action }: ControlCardProps) {
  return (
    <Card className={cn("control-panel", className)}>
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <CardTitle className="font-display text-lg tracking-tight">
            {title}
          </CardTitle>
          {action && (
            <div className="flex items-center gap-2">
              {action}
            </div>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {children}
      </CardContent>
    </Card>
  );
}