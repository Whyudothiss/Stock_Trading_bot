import os
import logging
from datetime import datetime, timedelta
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler
from pymongo import MongoClient
from bson import ObjectId
import asyncio
from dotenv import load_dotenv
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO

load_dotenv()

# Logging set Up
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

MONGO_URI = os.getenv("MONGO_URI")
BOT_Token = os.getenv("BOT_TOKEN")

if not MONGO_URI or not BOT_Token:
    raise ValueError("Please set MONGO_URI and BOT_TOKEN environment variables")

# MongoDB setup
try:
    from urllib.parse import quote_plus
    # If URI contains username and password, encode them
    if '@' in MONGO_URI:
        # Split the URI into parts
        prefix = MONGO_URI.split('://', 1)[0]
        rest = MONGO_URI.split('://', 1)[1]
        credentials = rest.split('@')[0]
        host_part = rest.split('@')[1]
        username = credentials.split(':')[0]
        password = credentials.split(':')[1]
        
        # URL encode the username and password
        encoded_username = quote_plus(username)
        encoded_password = quote_plus(password)
        
        # Reconstruct the URI
        MONGO_URI = f"{prefix}://{encoded_username}:{encoded_password}@{host_part}"

    # Connect with SSL settings for Atlas
    client = MongoClient(
        MONGO_URI,
        tlsAllowInvalidCertificates=True  # This bypasses certificate verification
    )
    # Test the connection
    client.admin.command('ping')
    print("Successfully connected to MongoDB!")
except Exception as e:
    print(f"Error connecting to MongoDB: {str(e)}")
    raise
db = client.portfoloio_tracker

# Collections
users_collection = db.users
transactions_collection = db.transactions
portfolio_values_collection = db.portfolio_values

(TICKER, QUANTITY, PRICE, TRIM_TYPE, TRIM_VALUE, 
 DELETE_CONFIRM, EDIT_SELECT, EDIT_FIELD, EDIT_VALUE) = range(9)

class PortfolioTracker():
    def __init__(self):
        self.user_data = {}
        
    def get_real_time_price(self, ticker):
        """Get real-time stock price from Yahoo Finance"""
        try:
            stock = yf.Ticker(ticker)
            current_price = stock.info.get('regularMarketPrice')
            return current_price if current_price else None
        except Exception as e:
            logger.error(f"Error fetching price for {ticker}: {str(e)}")
            return None

    def calculate_portfolio_value(self, user_id):
        """Calculate current portfolio value using real-time prices"""
        total_value = 0
        holdings = {}
        
        # Get all buy transactions
        transactions = list(transactions_collection.find({"user_id": user_id}).sort("timestamp", 1))
        
        for tx in transactions:
            ticker = tx['ticker']
            quantity = tx['quantity']
            if tx['action'] == 'buy':
                holdings[ticker] = holdings.get(ticker, 0) + quantity
            else:  # sell or trim
                holdings[ticker] = holdings.get(ticker, 0) - quantity
                
        # Calculate current value using real-time prices
        for ticker, shares in holdings.items():
            if shares > 0:
                current_price = self.get_real_time_price(ticker)
                if current_price:
                    total_value += shares * current_price
                    
        return total_value

    async def update_portfolio_value(self, user_id):
        """Update daily portfolio value in database"""
        current_value = self.calculate_portfolio_value(user_id)
        portfolio_values_collection.insert_one({
            "user_id": user_id,
            "value": current_value,
            "timestamp": datetime.now()
        })

    def generate_equity_curve(self, user_id):
        """Generate equity curve plot"""
        # Get historical portfolio values
        values = list(portfolio_values_collection.find(
            {"user_id": user_id}
        ).sort("timestamp", 1))
        
        if not values:
            return None
            
        # Create dataframe
        df = pd.DataFrame(values)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df['value'])
        plt.title('Portfolio Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Save plot to bytes buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        username = update.effective_user.username or update.effective_user.first_name

        # Ensure user is in the data dictionary
        if not users_collection.find_one({"user_id": user_id}):
            users_collection.insert_one({
                "user_id": user_id,
                "username": username,
                "create_at": datetime.now()
            })

        welcome_msg = """ 🎯 Welcome to myPortfolioTracker! 🎯

Your personal stock portfolio management bot. Here are the available commands:

📊 Portfolio Commands:
/buy - Record a stock purchase
/sell - Record a stock sale  
/trim - Partial sell (by % or quantity)
/current - View current holdings
/past - View sold positions with P&L
/pnl - Overall profit/loss summary
/portfolio - Quick portfolio overview
/value - Real-time portfolio value
/equity - Display equity curve

🛠 Management Commands:
/delete - Delete a transaction
/edit - Edit a transaction
/help - Show this help message

Ready to track your investments! Use /buy to record your first purchase."""


        await update.message.reply_text(welcome_msg)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.start(update, context)

    async def equity_curve(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Display the equity curve"""
        user_id = update.effective_user.id
        
        # Update current portfolio value
        await self.update_portfolio_value(user_id)
        
        # Generate equity curve
        plot_buffer = self.generate_equity_curve(user_id)
        
        if plot_buffer:
            current_value = self.calculate_portfolio_value(user_id)
            caption = f"📈 Your Portfolio Equity Curve\nCurrent Value: ${current_value:,.2f}"
            await update.message.reply_photo(plot_buffer, caption=caption)
        else:
            await update.message.reply_text("No portfolio data available yet. Start trading to see your equity curve!")

    async def value(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Display current portfolio value with real-time prices"""
        user_id = update.effective_user.id
        current_value = self.calculate_portfolio_value(user_id)
        
        # Get holdings with current prices
        holdings = {}
        transactions = list(transactions_collection.find({"user_id": user_id}).sort("timestamp", 1))
        
        for tx in transactions:
            ticker = tx['ticker']
            quantity = tx['quantity']
            if tx['action'] == 'buy':
                holdings[ticker] = holdings.get(ticker, 0) + quantity
            else:  # sell or trim
                holdings[ticker] = holdings.get(ticker, 0) - quantity
        
        message = "📊 Current Portfolio Value (Real-time)\n\n"
        
        for ticker, shares in holdings.items():
            if shares > 0:
                current_price = self.get_real_time_price(ticker)
                if current_price:
                    position_value = shares * current_price
                    message += f"{ticker}: {shares} shares @ ${current_price:.2f} = ${position_value:,.2f}\n"
        
        message += f"\n💰 Total Portfolio Value: ${current_value:,.2f}"
        await update.message.reply_text(message)


    # Buy transaction flow
    async def buy_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Start buy trasnsaction"""
        context.user_data['action'] = 'buy'
        await update.message.reply_text("Let's record a stock purchase!\n\n Whats the ticker symbol (e.g., AAPL):")
        return TICKER
    
    async def get_ticker(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Get ticker symbol"""
        ticker = update.message.text.strip().upper()
        context.user_data['ticker'] = ticker

        await update.message.reply_text(f"Stock: {ticker}\n\n How many shares did you buy?")
        return QUANTITY
    
    async def get_quantity(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        try:
            quantity = float(update.message.text)

            if quantity <= 0:
                await update.message.reply_text("Please enter a positive number")
                return QUANTITY

            context.user_data['quantity'] = quantity
            ticker = context.user_data['ticker']
            action = context.user_data['action']

            await update.message.reply_text(f"Stock: {ticker}\nQuantity: {quantity} shares\n\n What price did you {'buy' if action == 'buy' else 'sell'} at? (USD)")
            return PRICE
        
        except ValueError:
            await update.message.reply_text("Please enter a valid number")
            return QUANTITY
            
    async def get_price_and_execute(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        try:
            price = float(update.message.text)    
            if price <= 0:
                await update.message.reply_text("PLease enter a positive price")
                return PRICE

            user_id = update.effective_user.id
            ticker = context.user_data['ticker']
            quantity = context.user_data['quantity']
            action = context.user_data['action']
            total_value = quantity * price

            # Check for sell/trim whether user has enough shares
            if action in ['sell', 'trim']:
                current_position = self.get_current_positions(user_id, ticker)
                if current_position < quantity:
                    await update.message.reply_test(f"❌ Error: You only own {current_position} shares of {ticker}, "
                        f"but trying to {action} {quantity} shares.\n\nTransaction cancelled.")
                    return ConversationHandler.END

            transaction ={
                "user_id": user_id,
                "ticker": ticker,
                "action": action,
                "price": price,
                "quantity": quantity,
                "total_value": total_value,
                "timestamp": datetime.now()
            }

            transactions_collection.insert_one(transaction)

            action_emoji = "📈" if action == "buy" else "📉"
            confirmation = f"""
{action_emoji} Transaction Recorded!
Stock: {ticker}
Action: {action.upper()}
Quantity: {quantity} shares
Price: ${price:.2f}
Total: ${total_value:.2f}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

✅ Successfully added to your portfolio! 
           """
            await update.message.reply_text(confirmation)
            return ConversationHandler.END
        except ValueError:
            await update.message.reoly_text("Please enter a valid price)")
            return PRICE

    async def sell_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start sell transaction"""
        context.user_data['action'] = 'sell'
        await update.message.reply_text("📉 Let's record a stock sale!\n\nWhat's the ticker symbol?")
        return TICKER

    async def trim_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Start trim transaction"""
        await update.message.reply_text("✂️ Let's record a stock trim (partial sell)!\n\nWhat's the ticker symbol?")
        return TICKER

    async def get_trim_ticker(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Get ticker symbol for trim"""
        ticker = update.message.text.strip().upper()
        user_id = update.effective_user.id

        current_position = self.get_current_positions(user_id, ticker)
        if current_position <= 0:
            await update.message.reply_text(f"❌ You don't currently own any {ticker} shares.")
            return ConversationHandler.END
        
        context.user_data['ticker'] = ticker
        context.user_data['current_position'] = current_position

        keyboard = [['Percentage', 'Quantity']]
        reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)

        await update.message.reply_test(
            f"Stock: {ticker}\nCurrent position: {current_position} shares\n\n"
            "How would you like to trim?",
            reply_markup=reply_markup
        )
        return TRIM_TYPE
    
    async def get_trim_type(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        trim_type = update.message.text.strip().lower()
        if trim_type not in ['percentage', 'quantity']:
            await update.message.reply_text("Please choose either Percentage or Quantity")
            return TRIM_TYPE
        ticker = context.user_data['ticker']
        context.user_data['trim_type'] = trim_type
        current_position = context.user_data['current)position']

        if trim_type == 'percentage':
            await update.message.reply_test(
                f"Stock: {ticker}\nCurrent: {current_position} shares\n\n"
                "What percentage would you like to sell? (e.g., 25 for 25%)",
                reply_markup=ReplyKeyboardRemove()
            )
        else:
            await update.message.reply_text(
                f"Stock: {ticker}\nCurrent: {current_position} shares\n\n"
                "How many shares would you like to sell?",
                reply_markup=ReplyKeyboardRemove()
            )
        return TRIM_VALUE

    async def get_trim_value(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        try:
            value = float(update.message.text)
            trim_type = context.user_data['trim_type']
            ticker = context.user_data['ticker']
            current_position = context.user_data['current_position']

            # For percentage trim type
            if trim_type == 'percentage':
                if value <= 0 or value > 100:
                    await update.message.reply_text("Please enter a percentage between 0 and 100")
                    return TRIM_VALUE
                quantity = (value / 100) * current_position
            # For quantity trim type
            else:
                if value <= 0 or value > current_position:
                    await update.message.reply_text(f"Please enter a quantity between 0 and {current_position}")
                    return TRIM_VALUE
                quantity = value
            
            context.user_data['quantity'] = quantity
            context.user_data['action'] = 'trim'

            await update.message.reply_text(f"Stock: {ticker}\n"
                f"Selling: {quantity:.2f} shares ({(quantity/current_position)*100:.1f}%)\n"
                f"Remaining: {current_position - quantity:.2f} shares\n\n"
                "At what price are you selling? (USD)"
            )
            return PRICE
        except ValueError:
            await update.message.reply_text("Please enter a valid number")
            return TRIM_VALUE

    def get_current_positions(self, user_id, ticker):
        """Get current positions for a user and ticker"""
        pipeline = [
            {"$match": {"user_id": user_id, "ticker": ticker}},
            {"$group": {
                "_id": "$ticker",
                "total_bought": {"$sum": {"$cond": [{"$eq": ["$action", "buy"]}, "$quantity", 0]}},
                "total_sold": {"$sum": {"$cond": [{"$neq": ["$action", "buy"]}, "$quantity", 0]}}
            }}
        ]

        result = list(transactions_collection.aggregate(pipeline))
        if result:
            return result[0]["total_bought"] - result[0]["total_sold"]
        return 0
    
    def get_user_holdings(self, user_id):
        """Get all current holdings for a user"""
        pipeline = [
            {"$match": {"user_id": user_id}},
            {"$group": {
                "_id": "$ticker",
                "total_bought_qty": {"$sum": {"$cond": [{"$eq": ["$action", "buy"]}, "$quantity", 0]}},
                "total_sold_qty": {"$sum": {"$cond": [{"$neq": ["$action", "buy"]}, "$quantity", 0]}},
                "total_bought_value": {"$sum": {"$cond": [{"$eq": ["$action", "buy"]}, "$total_value", 0]}},
                "total_sold_value": {"$sum": {"$cond": [{"$neq": ["$action", "buy"]}, "$total_value", 0]}}
            }},    
            {"$addFields": {"$current_quantity": {"$subtract": ["total_bought_qty", "total_sold_qty"]}},
             "$current_cost": {"$subtract": ["$total_bought_value", {"multiply": [{"$divide": ["$total_bought_value", "$total_bought_qty"]}, "$total_sold_qty"]}]
            }},                   
            {"$match": {"$current_quantity": {"$gt": 0}}}
        ]

        results = list(transactions_collection.aggregate(pipeline))

        holdings = {}
        for result in results:
            ticker = result["_id"]
            # calculate weighted average cost for remaining shares
            avg_buy_price = result["total_bought_value"] / result["total_bought_qty"] if result["total_bought_qty"] > 0 else 0
            remaing_cost = avg_buy_price * result["$current_quantity"]

            holdings[ticker] = {
                "quantity": result["$current_quantity"],
                "total_cost": remaing_cost,
                "avg_cost": avg_buy_price
            }
        
        return holdings
    
    def get_past_positions_with_pnl(self, user_id):
        """Get past positions that were completely sold with P&L data"""
        pipeline = [
            {"$match": {"user_id": user_id}},
            {"$group": {
                "_id": "$ticker",
                "total_bought_qty": {"$sum": {"$cond": [{"$eq": ["$action", "buy"]}, "$quantity", 0]}},
                "total_sold_qty": {"$sum": {"$cond": [{"$neq": ["$action", "buy"]}, "$quantity", 0]}},
                "total_bought_value": {"$sum": {"$cond": [{"$eq": ["$action", "buy"]}, "$total_value", 0]}},
                "total_sold_value": {"$sum": {"$cond": [{"$neq": ["$action", "buy"]}, "$total_value", 0]}}
            }},
            {"$match": {
                "$and": [
                    {"$total_sold_qty": {"$gt": 0}},
                    {"$expr": {"$eq": ["$total_bought_qty", "$total_sold_qty"]}}
                ]
            }}
        ]
        
        results = list(transactions_collection.aggregate(pipeline))

        past_positions = {}

        for result in results:
            ticker = result["_id"]
            bought_qty = result["total_bought_qty"]
            sold_qty = min(result["total_sold_qty"], bought_qty)

            past_positions[ticker] = {
                "total_bought": result["total_bought_value"],
                "total_sold": (result["total_sold_value"] / result["total_sold_qty"]) * sold_qty if result["total_sold_qty"] > 0 else 0,
                "avg_buy_price": result["total_bought_value"] / bought_qty if bought_qty > 0 else 0,
                "avg_sell_price": result["total_sold_value"] / result["total_sold_qty"] if result["total_sold_qty"] > 0 else 0,
            }
        return past_positions
    
    async def current_holdings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show current holdings"""
        user_id = update.effective_user.id
        holdings = self.get_user_holdings(user_id)

        if not holdings:
            await update.message.reply_text("📊 You don't have any current holdings.")
            return
        
        message = "📊 Your Current Holdings:\n\n"
        total_invested = 0
        for ticker, data in holdings.items():
            avg_price = data["total_cost"] / data["quantity"]
            total_cost = data["total_cost"]
            total_invested += total_cost

            message += f"🏷 {ticker}\n"
            message += f"   Shares: {data['quantity']:.2f}\n"
            message += f"   Avg Price: ${avg_price:.2f}\n"
            message += f"   Total Cost: ${total_cost:.2f}\n\n"
        
        message += f"💰 Total Invested: ${total_invested:.2f}"

        await update.message.reply_text(message)

    async def past_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):   
        """Show past positions with P&L"""
        user_id = update.effective_user.id
        past_positions = self.get_past_positions_with_pnl(user_id)
        
        if not past_positions:
            await update.message.reply_text("📜 No past positions found.\n\nYou haven't sold any complete positions yet.")
            return
        
        message = "📜 Your Past Positions:\n\n"
        total_pnl = 0

        for ticker, data in past_positions.items():
            pnl = data["total_sold"] - data["total_bought"]
            pnl_pct = (pnl / data["total_bought"] * 100) if data["total_bought"] > 0 else 0
            total_pnl += pnl

            pnl_emoji = "📈" if pnl >= 0 else "📉"

            message += f"🏷 {ticker} {pnl_emoji}\n"
            message += f"   Avg Buy: ${data['avg_buy_price']:.2f}\n"
            message += f"   Avg Sell: ${data['avg_sell_price']:.2f}\n"
            message += f"   P&L: ${pnl:.2f} ({pnl_pct:+.1f}%)\n\n"

        pnl_emoji = "📈" if pnl >= 0 else "📉"    
        message += f"{pnl_emoji} Total Realized P&L: ${total_pnl:.2f}"

        await update.message.reply_text(message)

    async def pnl_summary(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show PnL summary"""
        user_id = update.effective_user.id
        past_poistions = self.get_past_positions_with_pnl(user_id)
        current_holdings = self.get_user_holdings(user_id)

        # Realised P&L from past positions
        realized_pnl = sum(
            (data["total_sold"] - data["total_bought"])
            for data in past_poistions.values()
        )

        # Current investments
        total_invested = sum(
            data["total_cost"] for data in current_holdings.values()
        )

        message = "💰 P&L SUMMARY 💰\n\n"
        message += f"📈 Realized P&L: ${realized_pnl:.2f}\n"
        message += f"💼 Current Invested: ${total_invested:.2f}\n"
        message += f"📊 Total Positions: {len(current_holdings)} current, {len(past_positions)} closed\n\n"

        if realized_pnl >= 0:
            message += "🎉 Great job! You're in profit territory!"
        else:
            message += "📊 Keep tracking! Every trade is a learning opportunity."

        await update.message.reply_text(message)

    async def portfolio_overview(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Quick portfolio overview"""
        user_id = update.effective_user.id
        current_holdings = self.get_user_holdings(user_id)
        past_positions = self.get_past_positions_with_pnl(user_id)

        total_invested = sum(data["total_cost"] for data in current_holdings.values())
        realized_pnl = sum((data["total_sold"] - data["total_bought"]) for data in past_positions.values())

        message = "🎯 PORTFOLIO OVERVIEW 🎯\n\n"
        message += f"💼 Active Positions: {len(current_holdings)}\n"
        message += f"💰 Total Invested: ${total_invested:.2f}\n"
        message += f"📈 Realized P&L: ${realized_pnl:.2f}\n"
        message += f"📜 Closed Positions: {len(past_positions)}\n\n"
        message += "Use /current for detailed holdings\nUse /pnl for full P&L breakdown"

        await update.message.reply_text(message)


    async def cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Cancel the conversation"""
        await update.message.reply_text(
            "❌ Operation cancelled.",
            reply_markup=ReplyKeyboardRemove()
        )
        return ConversationHandler.END

def main():
    tracker = PortfolioTracker()

    # Create application
    application = Application.builder().token(BOT_Token).build()

    # Buy conversation handler
    buy_handler = ConversationHandler(
        entry_points = [CommandHandler('buy', tracker.buy_start)],
        states = {
            TICKER: [MessageHandler(filters.TEXT & ~filters.COMMAND, tracker.get_ticker)],
            QUANTITY: [MessageHandler(filters.TEXT & ~filters.COMMAND, tracker.get_quantity)],
            PRICE: [MessageHandler(filters.TEXT & ~filters.COMMAND, tracker.get_price_and_execute)],     
        },
        fallbacks = [CommandHandler('cancel', tracker.cancel)],
    )

    # Sell conversation handler
    sell_handler = ConversationHandler(
        entry_points=[CommandHandler('sell', tracker.sell_start)],
        states={
            TICKER: [MessageHandler(filters.TEXT & ~filters.COMMAND, tracker.get_ticker)],
            QUANTITY: [MessageHandler(filters.TEXT & ~filters.COMMAND, tracker.get_quantity)],
            PRICE: [MessageHandler(filters.TEXT & ~filters.COMMAND, tracker.get_price_and_execute)],
        },
        fallbacks=[CommandHandler('cancel', tracker.cancel)],
    )

    # Trim conversation handler
    trim_handler = ConversationHandler(
        entry_points=[CommandHandler('trim', tracker.trim_start)],
        states={
            TICKER: [MessageHandler(filters.TEXT & ~filters.COMMAND, tracker.get_trim_ticker)],
            TRIM_TYPE: [MessageHandler(filters.TEXT & ~filters.COMMAND, tracker.get_trim_type)],
            TRIM_VALUE: [MessageHandler(filters.TEXT & ~filters.COMMAND, tracker.get_trim_value)],
            PRICE: [MessageHandler(filters.TEXT & ~filters.COMMAND, tracker.get_price_and_execute)],
        },
        fallbacks=[CommandHandler('cancel', tracker.cancel)],
    )

    # Add handlers
    application.add_handler(CommandHandler("start", tracker.start))
    application.add_handler(CommandHandler("help", tracker.help_command))
    application.add_handler(buy_handler)
    application.add_handler(sell_handler)
    application.add_handler(trim_handler)
    application.add_handler(CommandHandler("current", tracker.current_holdings))
    application.add_handler(CommandHandler("past", tracker.past_positions))
    application.add_handler(CommandHandler("pnl", tracker.pnl_summary))
    application.add_handler(CommandHandler("portfolio", tracker.portfolio_overview))
    application.add_handler(CommandHandler("value", tracker.value))
    application.add_handler(CommandHandler("equity", tracker.equity_curve))

    # Schedule daily portfolio value updates
    async def update_all_portfolios():
        while True:
            users = users_collection.find({})
            for user in users:
                await tracker.update_portfolio_value(user["user_id"])
            # Wait for 24 hours before next update
            await asyncio.sleep(24 * 60 * 60)

    # Start the background task
    application.job_queue.run_repeating(
        lambda ctx: asyncio.create_task(update_all_portfolios()),
        interval=timedelta(days=1)
    )

    # Configure port for Railway deployment
    port = int(os.getenv("PORT", 8080))
    print(f"🎯 myPortfolioTracker bot is starting on port {port}...")
    
    if os.getenv("RAILWAY_ENVIRONMENT"):
        print("Running on Railway - using webhooks")
        application.run_webhook(
            listen="0.0.0.0",
            port=port,
            webhook_url=f"https://{os.getenv('RAILWAY_PUBLIC_DOMAIN')}"
        )
    else:
        print("Running locally - using polling")
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    
if __name__ == '__main__':
    main()    